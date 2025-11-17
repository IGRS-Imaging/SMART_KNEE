# models/landmark_completion_model.py

import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_max

import config
from .encoder import Encoder
from .egnn_processor import EGNNProcessor   # <-- use our custom PyG-style EGNN
from .decoder import Decoder


class LandmarkCompletionModel(nn.Module):
    """
    End-to-end femur landmark completion model using custom EGNNProcessor.

    Forward(batch) returns:
        - coords_pred_flat: (N_total, 3)
        - pos_init_flat:    (N_total, 3)
    """

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
            edge_dim=config.EDGE_DIM,
            m_dim=config.HIDDEN_DIM,
            dropout=0.0,
        )

        self.decoder = Decoder(
            feat_dim=config.FEAT_DIM,
            hidden=config.HIDDEN_DIM,
            out_dim=3,
        )

        # If True, ground-truth coordinates for known nodes are kept fixed
        self.freeze_known = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_centroid(self, pos, known_mask, batch_idx, num_graphs):
        """
        Per-graph centroid using only known landmarks.
        pos:        (N_total, 3)
        known_mask: (N_total,)
        batch_idx:  (N_total,)
        """
        km = known_mask.float().unsqueeze(-1)    # (N,1)
        pos_known = pos * km                     # zero for unknown

        sum_pos = scatter_sum(pos_known, batch_idx, dim=0, dim_size=num_graphs)  # (B,3)
        count = scatter_sum(km, batch_idx, dim=0, dim_size=num_graphs)           # (B,1)

        centroid = sum_pos / count.clamp_min(1e-6)                               # (B,3)
        return centroid

    def _compute_scale(self, pos_norm, batch_idx, num_graphs):
        """
        Per-graph max radius scaling to stabilize training.
        pos_norm: (N_total, 3) centered coordinates.
        """
        norms = pos_norm.norm(dim=-1)  # (N,)
        r_max, _ = scatter_max(norms, batch_idx, dim=0, dim_size=num_graphs)  # (B,)
        r_max = r_max.clamp_min(1e-6)
        scale_nodes = r_max[batch_idx].unsqueeze(-1)  # (N,1)
        return scale_nodes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch):
        """
        batch fields used:
            batch.pos        : (N_total, 3)
            batch.known_mask : (N_total,)
            batch.side       : (B,)
            batch.edge_index : (2, E_total)
            batch.edge_attr  : (E_total, EDGE_DIM)
            batch.batch      : (N_total,) graph index per node

        Returns:
            coords_pred_flat: (N_total, 3)
            pos_init_flat:   (N_total, 3)
        """
        pos = batch.pos                      # (N_total,3)
        known_mask = batch.known_mask        # (N_total,)
        side = batch.side                    # (B,)
        edge_index = batch.edge_index        # (2,E_total)
        edge_attr = batch.edge_attr          # (E_total,1)
        batch_idx = batch.batch              # (N_total,)

        num_graphs = side.shape[0]

        # --------------------------------------------------------------
        # 1) Initialize unknown landmarks at centroid of known nodes
        # --------------------------------------------------------------
        centroid = self._compute_centroid(pos, known_mask, batch_idx, num_graphs)  # (B,3)
        centroid_nodes = centroid[batch_idx]                                       # (N_total,3)

        pos_init = pos.clone()
        pos_init[~known_mask] = centroid_nodes[~known_mask]

        # --------------------------------------------------------------
        # 2) Center + scale coordinates
        # --------------------------------------------------------------
        pos_norm = pos_init - centroid_nodes                                      # (N,3)
        scale_nodes = self._compute_scale(pos_norm, batch_idx, num_graphs)       # (N,1)
        pos_scaled = pos_norm / scale_nodes                                      # (N,3)

        # --------------------------------------------------------------
        # 3) Encode features
        # --------------------------------------------------------------
        feats = self.encoder(pos_scaled, known_mask, side, batch_idx)            # (N,F)

        # --------------------------------------------------------------
        # 4) EGNNProcessor: sparse message passing
        # --------------------------------------------------------------
        feats_p, pos_p = self.processor(
            feats, pos_scaled, edge_index, edge_attr, batch=batch_idx
        )                                                                         # (N,F),(N,3)

        # --------------------------------------------------------------
        # 5) Decode → coordinate offsets (still in normalized space)
        # --------------------------------------------------------------
        delta = self.decoder(feats_p)                                             # (N,3)
        coords_pred_scaled = pos_p + delta                                        # (N,3)

        # --------------------------------------------------------------
        # 6) Unscale + uncenter back to absolute coords
        # --------------------------------------------------------------
        coords_pred_norm = coords_pred_scaled * scale_nodes                       # (N,3)
        coords_pred = coords_pred_norm + centroid_nodes                           # (N,3)

        # --------------------------------------------------------------
        # 7) Optionally keep known nodes fixed
        # --------------------------------------------------------------
        if self.freeze_known:
            coords_pred = torch.where(
                known_mask.unsqueeze(-1),
                pos,          # ground truth for known nodes
                coords_pred,  # prediction for unknown nodes
            )

        # --------------------------------------------------------------
        # Return both prediction and initialized coords
        # --------------------------------------------------------------
        return coords_pred, pos_init
