# models/landmark_model.py
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
            edge_dim=config.EDGE_DIM, 
            m_dim=config.HIDDEN_DIM,
        )

        self.decoder = Decoder(
            feat_dim=config.FEAT_DIM,
            hidden=config.HIDDEN_DIM,
            out_dim=3,
        )

    def batch_kabsch(self, pred_pts, target_pts, mask):
        """
        Calculates rigid transformation (R, t) to map pred_pts[mask] -> target_pts[mask].
        """
        B, N, _ = pred_pts.shape
        device = pred_pts.device
        
        # Expand mask
        mask_float = mask.float().unsqueeze(-1) # (B, N, 1)
        
        # 1. Centroids of KNOWN nodes only
        # Add epsilon to count to avoid div by zero
        count = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
        
        pred_centroid = (pred_pts * mask_float).sum(dim=1, keepdim=True) / count
        target_centroid = (target_pts * mask_float).sum(dim=1, keepdim=True) / count
        
        # 2. Center the points
        P = (pred_pts - pred_centroid) * mask_float
        Q = (target_pts - target_centroid) * mask_float
        
        # 3. Covariance Matrix H = P^T * Q
        H = torch.matmul(P.transpose(1, 2), Q)
        
        # 4. SVD
        U, S, Vt = torch.linalg.svd(H)
        
        # 5. Rotation R = V * U^T
        # Note: torch.linalg.svd returns V^T (Vt), so we use Vt.T @ U.T
        # But standard Kabsch derivation often uses U @ V^T depending on definition of H.
        # With H = P^T Q: R = V U^T
        R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
        
        # 6. Reflection Correction (Ensure right-handed coordinate system)
        det = torch.det(R)
        # Create correction matrix
        I = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        I[:, 2, 2] = det # If det is -1, flip Z axis
        
        # Recalculate R
        R = torch.matmul(torch.matmul(Vt.transpose(1, 2), I), U.transpose(1, 2))
        
        # 7. Translation
        t = target_centroid - torch.matmul(pred_centroid, R)
        
        return R, t

    def forward(self, batch):
        pos = batch.pos
        known_mask = batch.known_mask
        side = batch.side
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        device = pos.device
        B = side.size(0)
        N = config.NUM_NODES

        # Reshape inputs
        pos_reshaped = pos.view(B, N, 3)
        known_mask_reshaped = known_mask.view(B, N)

        # --- 1. Pre-processing (Centering) ---
        km = known_mask.float().unsqueeze(-1)
        sum_pos = scatter_sum(pos * km, batch_idx, dim=0, dim_size=B)
        count = scatter_sum(km, batch_idx, dim=0, dim_size=B)
        centroid = sum_pos / count.clamp_min(1e-6)
        centroid_nodes = centroid[batch_idx]

        # Init unknown at centroid
        pos_init = pos.clone()
        pos_init[~known_mask] = centroid_nodes[~known_mask]
        pos_centered = pos_init - centroid_nodes

        # --- 2. Scaling ---
        sq_norms = (pos_centered ** 2).sum(dim=-1)
        mean_sq_norm = scatter_mean(sq_norms, batch_idx, dim=0, dim_size=B)
        rms = torch.sqrt(mean_sq_norm).clamp_min(1e-6)
        scale = rms
        scale_nodes = scale[batch_idx].unsqueeze(-1)
        pos_scaled = pos_centered / scale_nodes

        edge_batch_idx = batch_idx[edge_index[0]]
        edge_scale = scale[edge_batch_idx].unsqueeze(-1)
        edge_attr_scaled = edge_attr / edge_scale

        # --- 3. EGNN Inference ---
        feats = self.encoder(pos_scaled, known_mask, side, batch_idx)
        feats_out, pos_out = self.processor(feats, pos_scaled, edge_index, edge_attr_scaled)
        delta = self.decoder(feats_out)
        
        # Output in Scaled, Centered Space
        pos_pred_scaled = pos_out + delta

        # --- 4. Unscale to Original Size ---
        pos_pred_raw = pos_pred_scaled * scale_nodes + centroid_nodes
        
        # --- 5. HARD ANCHORING (Kabsch) ---
        # Force pred_known to align with gt_known
        pred_view = pos_pred_raw.view(B, N, 3)
        R, t = self.batch_kabsch(pred_view, pos_reshaped, known_mask_reshaped)
        
        # Apply transform to everything
        pos_pred_anchored = torch.matmul(pred_view, R) + t
        
        # Flatten for compatibility
        pos_pred_final = pos_pred_anchored.view(-1, 3)

        return pos_pred_final, pos_init