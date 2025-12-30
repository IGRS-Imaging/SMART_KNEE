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
            nn.Dropout(0.1),  # Add dropout
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
        )

        # Enhanced radial bias with more capacity
        self.radial_mlp = nn.Sequential(
            nn.Linear(1 + edge_dim, m_dim),  # Include edge_attr in radial computation
            nn.SiLU(),
            nn.Linear(m_dim, m_dim // 2),
            nn.SiLU(),
            nn.Linear(m_dim // 2, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + m_dim, feat_dim * 2),  # Wider layer
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        
        # Learnable gating for coordinate updates
        # This prevents the model from making wild coordinate changes
        self.coord_gate = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x, pos, edge_index, edge_attr):
        row, col = edge_index

        diff = pos[row] - pos[col]
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8

        # Edge message
        e_ij = torch.cat([x[row], x[col], dist, edge_attr], dim=-1)
        m_ij = self.edge_mlp(e_ij)

        # Enhanced radial coordinate update
        radial_input = torch.cat([dist, edge_attr], dim=-1)
        r = self.radial_mlp(radial_input)
        
        # Gating mechanism: learn how much to update each coordinate
        gate = self.coord_gate(m_ij)  # (num_edges, 1)
        
        # Apply gating to coordinate updates
        delta = gate * r * (diff / dist)

        pos_out = pos + scatter_add(delta, row, dim=0, dim_size=pos.size(0))

        # Node feature update (residual)
        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))
        x_out = x + self.node_mlp(torch.cat([x, m_i], dim=-1))

        return x_out, pos_out


class EGNNProcessor(nn.Module):
    def __init__(self, feat_dim, n_layers, edge_dim, m_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            EGNNLayer(feat_dim, edge_dim, m_dim)
            for _ in range(n_layers)
        ])
        
        # Add skip connections across layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feat_dim) for _ in range(n_layers)
        ])

    def forward(self, x, pos, edge_index, edge_attr):
        x_initial = x
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            x_residual = x
            x, pos = layer(x, pos, edge_index, edge_attr)
            
            # Layer normalization
            x = norm(x)
            
            # Skip connection every 2 layers
            if i % 2 == 1 and i > 0:
                x = x + x_initial
                x_initial = x
        
        return x, pos