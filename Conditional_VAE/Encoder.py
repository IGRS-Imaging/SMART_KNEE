import torch
import torch.nn as nn
from Config import COND_EMB_DIM, LATENT_DIM

def reparam(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

class Encoder(nn.Module):
    def __init__(self, point_feat_dim=3, c_dim=COND_EMB_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 + c_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, points, c):
        # points: (B, N, 3)
        p = self.point_mlp(points).max(dim=1).values  # (B, 256)
        h = torch.cat([p, c], dim=-1)
        h = self.fc(h)
        return self.fc_mu(h), self.fc_logvar(h)