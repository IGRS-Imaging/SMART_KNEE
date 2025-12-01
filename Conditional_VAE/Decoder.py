import torch
import torch.nn as nn
from Config import LATENT_DIM, COND_EMB_DIM, NUM_POINTS

class Decoder(nn.Module):
    """Decoder from latent space to point clouds with conditioning."""
    def __init__(self, latent_dim=LATENT_DIM, c_dim=COND_EMB_DIM, N=NUM_POINTS):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(latent_dim + c_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, N * 3)
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=-1)
        out = self.net(x).view(-1, self.N, 3)
        return out