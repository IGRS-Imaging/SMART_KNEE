import torch
import torch.nn as nn
from torch_scatter import scatter_add


class EGNNLayer(nn.Module):
    def __init__(self, feat_dim, edge_dim, m_dim):
        super().__init__()

        in_edge_dim = feat_dim * 2 + 1 + edge_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
        )

        # 🔥 Radial bias function (distance-aware)
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + m_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x, pos, edge_index, edge_attr):
        row, col = edge_index

        diff = pos[row] - pos[col]
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8

        # Edge message
        e_ij = torch.cat([x[row], x[col], dist, edge_attr], dim=-1)
        m_ij = self.edge_mlp(e_ij)

        # Radial coordinate update
        r = self.radial_mlp(dist)
        delta = r * (diff / dist)

        pos_out = pos + scatter_add(delta, row, dim=0, dim_size=pos.size(0))

        # Node feature update (residual)
        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))
        x_out = x + self.node_mlp(torch.cat([x, m_i], dim=-1))

        return x_out, pos_out


# ✅ THIS CLASS WAS MISSING — REQUIRED FOR IMPORT
class EGNNProcessor(nn.Module):
    def __init__(self, feat_dim, n_layers, edge_dim, m_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            EGNNLayer(feat_dim, edge_dim, m_dim)
            for _ in range(n_layers)
        ])

    def forward(self, x, pos, edge_index, edge_attr):
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
        return x, pos
