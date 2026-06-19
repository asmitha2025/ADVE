import numpy as np
import os
from typing import Optional
from adve.core.spatial_graph import SpatialGraph


class EmbeddingReconstructor:
    """
    Core hypothesis implementation:

        E(frame_t)  ≈  f( E_anchor, ΔG )

    Strategy
    --------
    Scene embedding is approximated using a trained MLP if weights exist.
    Otherwise, falls back to a closed-form weighted average of object embeddings,
    where each weight = (normalised area) × (stability factor).

    Stability factor: objects that moved a lot in ΔG get lower weight,
    because their anchor embedding is stale.

    Final result is blended with the raw anchor frame embedding based on
    total delta magnitude, so that for nearly-static scenes we stay close
    to the anchor embedding.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = "cpu"  # CPU is highly robust and very fast for this small MLP

        if not model_path:
            return

        paths_to_check = [
            model_path,
            os.path.join("adve_v2", model_path) if not os.path.isabs(model_path) else None,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", model_path) if not os.path.isabs(model_path) else None
        ]

        for path in paths_to_check:
            if path and os.path.exists(path):
                try:
                    import torch
                    import torch.nn as nn

                    class ReconstructionMLP(nn.Module):
                        def __init__(self, clip_dim=512, delta_dim=128, hidden_dim=512):
                            super().__init__()
                            input_dim = clip_dim + clip_dim + delta_dim
                            self.net = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.GELU(),
                                nn.Dropout(0.1),
                                nn.Linear(hidden_dim, hidden_dim // 2),
                                nn.LayerNorm(hidden_dim // 2),
                                nn.GELU(),
                                nn.Dropout(0.1),
                                nn.Linear(hidden_dim // 2, clip_dim),
                            )
                            self.residual_weight = nn.Parameter(torch.tensor(0.1))

                        def forward(self, anchor_emb, object_pool, delta_vec):
                            x = torch.cat([anchor_emb, object_pool, delta_vec], dim=-1)
                            delta_pred = self.net(x)
                            out = anchor_emb + self.residual_weight * delta_pred
                            return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

                    self.model = ReconstructionMLP()
                    self.model.load_state_dict(torch.load(path, map_location=self.device))
                    self.model.eval()
                    print(f"[EmbeddingReconstructor] Loaded ReconstructionMLP from {path}")
                    break
                except Exception as e:
                    print(f"[EmbeddingReconstructor] Failed to load model weights from {path}: {e}")

    def delta_to_vector(self, delta: dict, max_pairs: int = 32) -> np.ndarray:
        """Convert ΔG dict to fixed-size vector for MLP input."""
        vec = np.zeros(max_pairs * 4, dtype=np.float32)
        for i, (pair, rd) in enumerate(delta.get("relation_deltas", {}).items()):
            if i >= max_pairs:
                break
            base = i * 4
            vec[base]     = rd["delta_distance"]
            vec[base + 1] = rd["delta_angle"]
            vec[base + 2] = rd["delta_size_ratio"]
            vec[base + 3] = rd["magnitude"]
        return vec

    def pool_object_embeddings(
        self, objects: dict, max_objects: int = 8
    ) -> np.ndarray:
        """Pool per-object embeddings to fixed 512-d vector."""
        valid = [
            (obj.area, obj.embedding)
            for obj in objects.values()
            if obj.embedding is not None
        ]
        if not valid:
            return np.zeros(512, dtype=np.float32)

        valid.sort(key=lambda x: -x[0])  # sort by area, largest first
        valid = valid[:max_objects]

        weights = np.array([a for a, _ in valid], dtype=np.float32)
        weights /= weights.sum() + 1e-8
        embs = np.array([e for _, e in valid])

        return (weights[:, None] * embs).sum(axis=0).astype(np.float32)

    def reconstruct(
        self,
        anchor_graph:     SpatialGraph,
        current_graph:    SpatialGraph,
        delta:            dict,
        anchor_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Returns a normalised 512-d embedding approximating the current frame.
        """
        # If MLP is loaded, attempt learned reconstruction
        if self.model is not None:
            try:
                import torch
                delta_vec = self.delta_to_vector(delta)
                object_pool = self.pool_object_embeddings(current_graph.objects)

                anchor_t = torch.tensor(anchor_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                pool_t = torch.tensor(object_pool, dtype=torch.float32).unsqueeze(0).to(self.device)
                delta_t = torch.tensor(delta_vec[:128], dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred_t = self.model(anchor_t, pool_t, delta_t)
                    reconstructed = pred_t.squeeze(0).cpu().numpy()
                return reconstructed.astype(np.float32)
            except Exception as e:
                print(f"[EmbeddingReconstructor] MLP inference failed: {e}. Falling back to weighted average.")

        # Fallback to closed-form weighted average
        valid_objs = {
            oid: obj
            for oid, obj in current_graph.objects.items()
            if obj.embedding is not None
        }

        # Fallback: if no objects tracked, return anchor embedding unchanged
        if not valid_objs:
            return anchor_embedding.copy()

        embeddings: list = []
        weights:    list = []

        total_area = sum(obj.area for obj in valid_objs.values()) + 1e-8

        for oid, obj in valid_objs.items():
            area_weight = obj.area / total_area

            # Penalise weight of objects that moved a lot
            positional_change = 0.0
            for pair, rel_delta in delta.get("relation_deltas", {}).items():
                if oid in pair:
                    positional_change += rel_delta["magnitude"]

            # Stability ∈ (0, 1]: high = object barely moved
            stability = 1.0 / (1.0 + positional_change)

            embeddings.append(obj.embedding)
            weights.append(area_weight * stability)

        weights_arr = np.array(weights, dtype=np.float32)
        weights_arr /= weights_arr.sum() + 1e-8

        object_blend = np.sum(
            [w * e for w, e in zip(weights_arr, embeddings)],
            axis=0
        ).astype(np.float32)

        # Blend: when delta ≈ 0 → trust anchor; when delta is large → trust object blend
        blend = float(np.clip(delta.get("total_magnitude", 0.0), 0.0, 1.0))
        reconstructed = (1.0 - blend) * anchor_embedding + blend * object_blend

        # Normalise to unit sphere (CLIP convention)
        norm = np.linalg.norm(reconstructed)
        if norm > 1e-8:
            reconstructed /= norm

        return reconstructed.astype(np.float32)
