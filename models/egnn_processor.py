# models/egnn_processor.py

import torch
import torch.nn as nn
from torch_scatter import scatter_add


class EGNNLayer(nn.Module):
    """
    E(n)-Equivariant GNN layer (sparse, batched).
    x: (N_total,F), pos: (N_total,3), edge_index, edge_attr
    """

    def __init__(self, feat_dim, edge_dim=1, m_dim=64, dropout=0.0):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1 + edge_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + m_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, edge_index, edge_attr=None, batch=None):
        """
        x:         (N_total, F)
        pos:       (N_total, 3)
        edge_index:(2, E)
        edge_attr: (E, D_e)
        batch:     (N_total,) graph idx (kept for future masks, unused here)
        """
        row, col = edge_index  # messages from col -> row

        rel_pos = pos[row] - pos[col]                      # (E, 3)
        dist2 = (rel_pos ** 2).sum(dim=-1, keepdim=True)   # (E, 1)

        edge_inputs = [x[row], x[col], dist2]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)

        e_ij = torch.cat(edge_inputs, dim=-1)              # (E, 2F+1+De)
        m_ij = self.edge_mlp(e_ij)                         # (E, m_dim)

        # ---- Coordinate update ----
        coord_w = self.coord_mlp(m_ij)                     # (E,1)
        coord_w = coord_w.clamp(-1.0, 1.0)                 # clamp for stability
        delta_ij = coord_w * rel_pos                       # (E,3)
        delta_pos = scatter_add(delta_ij, row, dim=0, dim_size=x.size(0))
        pos_out = pos + delta_pos

        # ---- Feature update ----
        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))  # (N,m_dim)
        feat_input = torch.cat([x, m_i], dim=-1)
        updated = self.node_mlp(feat_input)

        x_out = x + self.dropout(self.norm(updated))       # residual connection

        return x_out, pos_out


class EGNNProcessor(nn.Module):
    """
    Stack of EGNNLayers.
    """

    def __init__(self, feat_dim=128, n_layers=4, edge_dim=1, m_dim=64, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(feat_dim, edge_dim=edge_dim, m_dim=m_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, pos, edge_index, edge_attr=None, batch=None):
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr, batch=batch)
        return x, pos
