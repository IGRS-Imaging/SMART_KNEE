import torch
import torch.nn as nn
import numpy as np
import config
from .encoder import Encoder
from .egnn_processor import EGNNProcessor

class LandmarkCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Learnable Template
        try:
            mean_np = np.load(config.MEAN_SHAPE_PATH)
            mean_tensor = torch.tensor(mean_np, dtype=torch.float32)
            self.mean_canonical_shape = nn.Parameter(mean_tensor)
            print("✅ Initialized Learnable Template from file.")
        except:
            print("⚠️ Warning: Mean shape file not found. Initializing random.")
            self.mean_canonical_shape = nn.Parameter(torch.randn(config.NUM_NODES, 3))

        self.register_buffer("reflection_matrix", torch.tensor(np.diag([-1, 1, 1]), dtype=torch.float32))

        self.encoder = Encoder(config.FEAT_DIM, config.HIDDEN_DIM, use_coords=True)
        
        self.processor = EGNNProcessor(
            feat_dim=config.FEAT_DIM,
            n_layers=config.GNN_LAYERS,
            edge_dim=config.EDGE_DIM, 
            m_dim=config.HIDDEN_DIM,
        )

    def batch_kabsch_similarity(self, pred_pts, target_pts, mask):
        """
        Aligns pred_pts to target_pts using Similiarity Transform (Rotation + Translation + Scale).
        """
        B, N, _ = pred_pts.shape
        device = pred_pts.device
        mask_float = mask.float().unsqueeze(-1)
        
        # 1. Centroids
        count = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pred_c = (pred_pts * mask_float).sum(dim=1, keepdim=True) / count
        target_c = (target_pts * mask_float).sum(dim=1, keepdim=True) / count
        
        # Center the points
        P = (pred_pts - pred_c) * mask_float
        Q = (target_pts - target_c) * mask_float
        
        # 2. Scale Estimation
        # p_var: (B, 1), q_var: (B, 1)
        p_var = (P ** 2).sum(dim=-1).sum(dim=1, keepdim=True)
        q_var = (Q ** 2).sum(dim=-1).sum(dim=1, keepdim=True)
        s = torch.sqrt(q_var / p_var.clamp_min(1e-6)) # Shape (B, 1)
        
        # 3. Rotation (Kabsch)
        # Fix: Reshape s to (B, 1, 1) for broadcasting against P (B, N, 3)
        P_scaled = P * s.unsqueeze(-1) 
        
        H = torch.matmul(P_scaled.transpose(1, 2), Q)
        U, S, Vt = torch.linalg.svd(H)
        
        # Reflection Correction
        R_candidate = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
        det = torch.det(R_candidate)
        
        I = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        flip_mask = (det < 0).long()
        I[:, 2, 2] = 1 - 2 * flip_mask 
        
        R = torch.matmul(torch.matmul(U, I), Vt)
        
        # 4. Translation
        # t = target_c - s * (R @ pred_c)
        # s needs to be (B, 1, 1) here too, but PyTorch handles scalar-matrix mul automatically if order is right.
        # Ideally: s.view(B, 1, 1) * ...
        t = target_c - s.unsqueeze(-1) * torch.matmul(pred_c, R)
        
        return s.unsqueeze(-1), R, t # Return s as (B, 1, 1) for safety

    def align_template(self, batch_size, pos_target, known_mask, side):
        template = self.mean_canonical_shape.unsqueeze(0).repeat(batch_size, 1, 1)
        
        is_left = (side == 0)
        if is_left.any():
            reflected_template = torch.matmul(template, self.reflection_matrix)
            template[is_left] = reflected_template[is_left]

        template_detached = template.detach()
        
        # Calculate Similarity Transform (Scale + Rot + Trans)
        # s comes back as (B, 1, 1)
        s, R, t = self.batch_kabsch_similarity(template_detached, pos_target, known_mask)
        
        # Apply: (Template * s) @ R + t
        aligned_template = s * torch.matmul(template, R) + t
        
        return aligned_template

    def get_anchor_centroid(self, pos, mask):
        mask_float = mask.float().unsqueeze(-1)
        count = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
        centroid = (pos * mask_float).sum(dim=1, keepdim=True) / count
        return centroid

    def forward(self, batch):
        pos_gt = batch.pos
        known_mask = batch.known_mask
        side = batch.side
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        B = side.size(0)
        N = config.NUM_NODES
        pos_gt_reshaped = pos_gt.view(B, N, 3)
        known_mask_reshaped = known_mask.view(B, N)

        # 1. Initialize Hybrid Input
        # Align template (with SCALE correction) to known GT
        pos_template_aligned = self.align_template(B, pos_gt_reshaped, known_mask_reshaped, side)
        
        # Stitch: GT for Known, Scaled Template for Unknown
        pos_init = pos_template_aligned.clone()
        pos_init[known_mask_reshaped] = pos_gt_reshaped[known_mask_reshaped]
        
        # 2. Anchor-Based Normalization
        anchor_centroid = self.get_anchor_centroid(pos_init, known_mask_reshaped)
        anchor_centroid_expanded = anchor_centroid.repeat(1, N, 1).view(-1, 3)
        
        pos_centered = (pos_init.view(-1, 3) - anchor_centroid_expanded) / config.GLOBAL_SCALE
        edge_attr_scaled = edge_attr / config.GLOBAL_SCALE

        # 3. GNN Processing
        feats = self.encoder(pos_centered, known_mask, side, batch_idx)
        feats_out, pos_out_centered = self.processor(feats, pos_centered, edge_index, edge_attr_scaled)
        
        # 4. Residual Delta Prediction & Structural Lock
        delta = pos_out_centered - pos_centered
        inv_mask = (~known_mask).float().unsqueeze(-1)
        delta_masked = delta.view(B, N, 3) * inv_mask.view(B, N, 1)
        
        pos_pred_centered = pos_centered.view(B, N, 3) + delta_masked
        
        # 5. Unscale
        pos_pred_final = (pos_pred_centered * config.GLOBAL_SCALE) + anchor_centroid
        
        return pos_pred_final.view(-1, 3), pos_init.view(-1, 3)