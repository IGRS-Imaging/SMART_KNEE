# models/landmark_completion_model.py
import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_max

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
            edge_dim=0,       # no edge_attr
            m_dim=config.HIDDEN_DIM,
        )

        self.decoder = Decoder(
            feat_dim=config.FEAT_DIM,
            hidden=config.HIDDEN_DIM,
            out_dim=3,
        )

    # -----------------------------------------------------
    def forward(self, batch):
        pos = batch.pos
        known_mask = batch.known_mask
        side = batch.side
        edge_index = batch.edge_index
        batch_idx = batch.batch

        device = pos.device
        B = side.size(0)

        # ---- centroid (known only) ----
        km = known_mask.float().unsqueeze(-1)
        sum_pos = scatter_sum(pos * km, batch_idx, dim=0, dim_size=B)
        count = scatter_sum(km, batch_idx, dim=0, dim_size=B)
        centroid = sum_pos / count.clamp_min(1e-6)
        centroid_nodes = centroid[batch_idx]

        # ---- initialize unknown at centroid ----
        pos_init = pos.clone()
        pos_init[~known_mask] = centroid_nodes[~known_mask]

        # ---- normalize by subtracting centroid ----
        pos_centered = pos_init - centroid_nodes

        # ---- scale to unit-ish radius ----
        norms = pos_centered.norm(dim=-1)
        r_max, _ = scatter_max(norms, batch_idx, dim=0, dim_size=B)
        scale = r_max.clamp_min(1e-6)
        scale_nodes = scale[batch_idx].unsqueeze(-1)
        pos_scaled = pos_centered / scale_nodes

        # ---- Encode node features ----
        feats = self.encoder(pos_scaled, known_mask, side, batch_idx)

        # ---- EGNN ----
        feats_out, pos_out = self.processor(feats, pos_scaled, edge_index)

        # ---- decode ----
        delta = self.decoder(feats_out)
        pos_pred_scaled = pos_out + delta

        # ---- unscale + uncenter ----
        pos_pred = pos_pred_scaled * scale_nodes + centroid_nodes

        return pos_pred, pos_init
