import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean

import config
from .encoder import Encoder
from .egnn_processor import EGNNProcessor
from .decoder import Decoder


class LandmarkCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(
            feat_dim=config.FEAT_DIM,
            hidden=config.HIDDEN_DIM,
            use_coords=True,
        )

        self.processor = EGNNProcessor(
            feat_dim=config.FEAT_DIM,
            n_layers=config.GNN_LAYERS,
            edge_dim=config.EDGE_DIM, # Now 1 (distance)
            m_dim=config.HIDDEN_DIM,
        )

        self.decoder = Decoder(
            feat_dim=config.FEAT_DIM,
            hidden=config.HIDDEN_DIM,
            out_dim=3,
        )

    def forward(self, batch):
        pos = batch.pos
        known_mask = batch.known_mask
        side = batch.side
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        device = pos.device
        B = side.size(0)

        # -------------------------------------------------------
        # 1. Centroid Calculation (using known nodes only)
        # -------------------------------------------------------
        km = known_mask.float().unsqueeze(-1)
        sum_pos = scatter_sum(pos * km, batch_idx, dim=0, dim_size=B)
        count = scatter_sum(km, batch_idx, dim=0, dim_size=B)
        centroid = sum_pos / count.clamp_min(1e-6)
        centroid_nodes = centroid[batch_idx]

        # Initialize unknown nodes at centroid
        pos_init = pos.clone()
        pos_init[~known_mask] = centroid_nodes[~known_mask]
        
        # Centering
        pos_centered = pos_init - centroid_nodes

        # -------------------------------------------------------
        # 2. ROBUST SCALING (RMS) - Replaces r_max
        # -------------------------------------------------------
        # Calculate standard deviation of distances from centroid
        sq_norms = (pos_centered ** 2).sum(dim=-1)
        # Average squared norm per graph
        mean_sq_norm = scatter_mean(sq_norms, batch_idx, dim=0, dim_size=B)
        rms = torch.sqrt(mean_sq_norm).clamp_min(1e-6)
        
        scale = rms # RMS scale
        scale_nodes = scale[batch_idx].unsqueeze(-1)
        
        pos_scaled = pos_centered / scale_nodes

        # -------------------------------------------------------
        # 3. Network Flow
        # -------------------------------------------------------
        # Also scale the target edge attributes for the network input
        # edge_attr is (Num_edges, 1). We need the scale of the graph associated with the edge
        # Get batch_idx for edges (using source node)
        edge_batch_idx = batch_idx[edge_index[0]]
        edge_scale = scale[edge_batch_idx].unsqueeze(-1)
        edge_attr_scaled = edge_attr / edge_scale

        feats = self.encoder(pos_scaled, known_mask, side, batch_idx)
        
        # Pass scaled edge attributes to EGNN
        feats_out, pos_out = self.processor(feats, pos_scaled, edge_index, edge_attr_scaled)

        delta = self.decoder(feats_out)
        pos_pred_scaled = pos_out + delta

        # -------------------------------------------------------
        # 4. Unscaling
        # -------------------------------------------------------
        pos_pred = pos_pred_scaled * scale_nodes + centroid_nodes

        return pos_pred, pos_init