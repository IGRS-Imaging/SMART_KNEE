#model/egnn_processor.py
import torch
import torch.nn as nn
from torch_scatter import scatter_add

class EGNNLayer(nn.Module):
    def __init__(self, feat_dim, edge_dim, m_dim):
        super().__init__()
        
        # Input to edge_mlp: [h_i, h_j, dist, edge_attr]
        in_edge_dim = feat_dim * 2 + 1 + edge_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.LayerNorm(m_dim) # Stabilization
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1) # Outputs a scalar weight
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + m_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim) # Stabilization
        )

    def forward(self, x, pos, edge_index, edge_attr):
        row, col = edge_index

        # 1. Calculate Radial Distance & Direction
        diff = pos[row] - pos[col]
        # Epsilon added to prevent division by zero
        dist = diff.norm(dim=-1, keepdim=True) + 1e-8 
        
        # 2. Edge Features: [Node_i, Node_j, Distance, Edge_Attr]
        # We use 'dist' (linear) instead of 'dist^2' for better gradient scaling
        e_ij = torch.cat([x[row], x[col], dist, edge_attr], dim=-1)
        
        # 3. Compute Message
        m_ij = self.edge_mlp(e_ij)

        # 4. Update Coordinates (STABILITY FIX)
        # Predict a scalar weight 'trans'
        trans = self.coord_mlp(m_ij)
        
        # Clamp the MAGNITUDE of the step, not the position itself
        # This allows movement but prevents single-step explosions
        trans = torch.clamp(trans, min=-5.0, max=5.0)
        
        # Normalize the direction vector: diff / dist
        # delta = weight * direction
        delta = trans * (diff / dist) 
        
        # Scatter add the updates
        pos_out = pos + scatter_add(delta, row, dim=0, dim_size=pos.size(0))

        # 5. Update Node Features (Residual Connection)
        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))
        x_out = x + self.node_mlp(torch.cat([x, m_i], dim=-1))

        return x_out, pos_out


class EGNNProcessor(nn.Module):
    def __init__(self, feat_dim, n_layers, edge_dim, m_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(feat_dim, edge_dim, m_dim) for _ in range(n_layers)
        ])

    def forward(self, x, pos, edge_index, edge_attr):
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
        return x, pos