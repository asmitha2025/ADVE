import numpy as np
from typing import Optional
from spatial_graph import SpatialGraph


class EmbeddingReconstructor:
    """
    Core hypothesis implementation:

        E(frame_t)  ≈  f( E_anchor, ΔG )

    Strategy
    --------
    Scene embedding is approximated as a weighted average of object embeddings,
    where each weight = (normalised area) × (stability factor).

    Stability factor: objects that moved a lot in ΔG get lower weight,
    because their anchor embedding is stale.

    Final result is blended with the raw anchor frame embedding based on
    total delta magnitude, so that for nearly-static scenes we stay close
    to the anchor embedding.
    """

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
            for pair, rel_delta in delta["relation_deltas"].items():
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
        blend = float(np.clip(delta["total_magnitude"], 0.0, 1.0))
        reconstructed = (1.0 - blend) * anchor_embedding + blend * object_blend

        # Normalise to unit sphere (CLIP convention)
        norm = np.linalg.norm(reconstructed)
        if norm > 1e-8:
            reconstructed /= norm

        return reconstructed.astype(np.float32)
