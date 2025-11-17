# models/decoder.py

import torch.nn as nn


class Decoder(nn.Module):
    """
    Map node features to coordinate offsets.
    """

    def __init__(self, feat_dim=128, hidden=128, out_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feats):
        return self.mlp(feats)  # (N_total, 3)
