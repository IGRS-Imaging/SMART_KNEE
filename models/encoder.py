# models/encoder.py

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Build per-node features from:
      - known_mask (1 dim per node)
      - side (graph-level 0/1, embedded and broadcast to nodes)
      - normalized coordinates (pos)
    Works on flat tensors:
      pos:        (N_total, 3)
      known_mask: (N_total,)
      side:       (B,)
      batch_idx:  (N_total,)
    """

    def __init__(
        self,
        feat_dim: int = 128,
        hidden: int = 128,
        side_embed_dim: int = 16,
        use_coords: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_coords = use_coords

        self.side_embed = nn.Embedding(2, side_embed_dim)

        in_dim = 1 + side_embed_dim + (3 if use_coords else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim),
            nn.SiLU(),
        )

    def forward(self, pos, known_mask, side, batch_idx):
        """
        pos:        (N_total, 3)
        known_mask: (N_total,)
        side:       (B,)
        batch_idx:  (N_total,)
        """
        km = known_mask.float().unsqueeze(-1)  # (N,1)

        side_emb = self.side_embed(side)       # (B, sdim)
        side_per_node = side_emb[batch_idx]    # (N, sdim)

        parts = [km, side_per_node]
        if self.use_coords:
            parts.append(pos)

        inp = torch.cat(parts, dim=-1)         # (N, in_dim)
        feats = self.mlp(inp)                  # (N, feat_dim)
        return feats
