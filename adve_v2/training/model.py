import torch
import torch.nn as nn


class ReconstructionMLP(nn.Module):
    """
    Learns to predict E(frame_t) from (E_anchor, ΔG_vector, object_pool).
    Replaces the closed-form weighted average in reconstructor.py.
    
    Once trained, drop this into reconstructor.py as a replacement.
    Expected improvement: 0.9484 → 0.97+ cosine sim.
    """

    def __init__(
        self,
        clip_dim:   int = 512,
        delta_dim:  int = 128,   # 32 pairs × 4 features
        hidden_dim: int = 512,
    ):
        super().__init__()
        input_dim = clip_dim + clip_dim + delta_dim  # anchor + pool + delta

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

        # Residual: start close to anchor, learn the delta
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        anchor_emb:  torch.Tensor,   # (B, 512)
        object_pool: torch.Tensor,   # (B, 512)
        delta_vec:   torch.Tensor,   # (B, 128)
    ) -> torch.Tensor:

        x = torch.cat([anchor_emb, object_pool, delta_vec], dim=-1)
        delta_pred = self.net(x)

        # Residual connection: output = anchor + learned_delta
        out = anchor_emb + self.residual_weight * delta_pred

        # Normalize to unit sphere (CLIP convention)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


class ReconstructionLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine_loss = 1.0 - (pred * target).sum(dim=-1).mean()
        mse_loss    = ((pred - target) ** 2).sum(dim=-1).mean()
        return cosine_loss + 0.1 * mse_loss
