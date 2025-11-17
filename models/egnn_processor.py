# models/egnn_processor.py
import torch
import torch.nn as nn
from torch_scatter import scatter_add


class EGNNLayer(nn.Module):
    def __init__(self, feat_dim, edge_dim, m_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim*2 + 1, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + m_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x, pos, edge_index):
        row, col = edge_index

        rel = pos[row] - pos[col]
        dist2 = (rel**2).sum(-1, keepdim=True)

        e_ij = torch.cat([x[row], x[col], dist2], dim=-1)
        m_ij = self.edge_mlp(e_ij)

        w = self.coord_mlp(m_ij)
        delta = w * rel
        pos_out = pos + scatter_add(delta, row, dim=0, dim_size=pos.size(0))

        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))
        x_out = self.node_mlp(torch.cat([x, m_i], dim=-1))

        return x_out, pos_out


class EGNNProcessor(nn.Module):
    def __init__(self, feat_dim, n_layers, edge_dim, m_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(feat_dim, edge_dim, m_dim) for _ in range(n_layers)
        ])

    def forward(self, x, pos, edge_index):
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index)
        return x, pos
