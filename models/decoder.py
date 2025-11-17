# models/decoder.py
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, feat_dim, hidden, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)
