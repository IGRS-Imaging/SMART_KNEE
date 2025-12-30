import torch
import torch.nn as nn
import numpy as np
import config
from .encoder import Encoder
from .egnn_processor import EGNNProcessor

class LandmarkCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CRITICAL FIX: Template should NOT be learnable
        # It's immediately overwritten by Kabsch alignment, so gradients are wasted
        try:
            mean_np = np.load(config.MEAN_SHAPE_PATH)
            mean_tensor = torch.tensor(mean_np, dtype=torch.float32)
            # Register as buffer (not parameter) - no gradients
            self.register_buffer('mean_canonical_shape', mean_tensor)
            print("✅ Initialized Fixed Template from file (non-learnable).")
        except:
            print("⚠️ Warning: Mean shape file not found. Initializing random.")
            mean_tensor = torch.randn(config.NUM_NODES, 3)
            self.register_buffer('mean_canonical_shape', mean_tensor)

        self.encoder = Encoder(config.FEAT_DIM, config.HIDDEN_DIM, use_coords=True)
        
        self.processor = EGNNProcessor(
            feat_dim=config.FEAT_DIM,
            n_layers=config.GNN_LAYERS,
            edge_dim=config.EDGE_DIM, 
            m_dim=config.HIDDEN_DIM,
        )
        
        # NEW: Confidence prediction head
        # Predict per-node confidence scores to identify problematic predictions
        self.confidence_head = nn.Sequential(
            nn.Linear(config.FEAT_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def batch_kabsch_similarity(self, pred_pts, target_pts, mask):
        """
        Aligns pred_pts to target_pts using Similarity Transform.
        """
        B, N, _ = pred_pts.shape
        device = pred_pts.device
        mask_float = mask.float().unsqueeze(-1)
        
        # Centroids
        count = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pred_c = (pred_pts * mask_float).sum(dim=1, keepdim=True) / count
        target_c = (target_pts * mask_float).sum(dim=1, keepdim=True) / count
        
        P = (pred_pts - pred_c) * mask_float
        Q = (target_pts - target_c) * mask_float
        
        # Scale
        p_var = (P ** 2).sum(dim=-1).sum(dim=1, keepdim=True)
        q_var = (Q ** 2).sum(dim=-1).sum(dim=1, keepdim=True)
        s = torch.sqrt(q_var / p_var.clamp_min(1e-6))
        
        # Rotation
        P_scaled = P * s.unsqueeze(-1)
        H = torch.matmul(P_scaled.transpose(1, 2), Q)
        U, S, Vt = torch.linalg.svd(H)
        
        # Reflection Check
        R = torch.matmul(U, Vt)
        det = torch.det(R)
        flip_mask = (det < 0).float().view(B, 1, 1)
        
        # Correct Reflection
        Vt_flipped = Vt.clone()
        Vt_flipped[:, 2, :] *= -1
        R_flipped = torch.matmul(U, Vt_flipped)
        
        R = torch.where(flip_mask > 0, R_flipped, R)

        # Translation
        t = target_c - s.unsqueeze(-1) * torch.matmul(pred_c, R)
        
        return s.unsqueeze(-1), R, t

    def align_template(self, batch_size, pos_target, known_mask):
        # Use fixed template (no learning)
        template = self.mean_canonical_shape.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Align template to input
        s, R, t = self.batch_kabsch_similarity(template, pos_target, known_mask)
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

        # 1. Align Template
        pos_template_aligned = self.align_template(B, pos_gt_reshaped, known_mask_reshaped)
        
        # 2. Stitch & Center
        pos_init = pos_template_aligned.clone()
        pos_init[known_mask_reshaped] = pos_gt_reshaped[known_mask_reshaped]
        
        anchor_centroid = self.get_anchor_centroid(pos_init, known_mask_reshaped)
        anchor_centroid_expanded = anchor_centroid.repeat(1, N, 1).view(-1, 3)
        
        pos_centered = (pos_init.view(-1, 3) - anchor_centroid_expanded) / config.GLOBAL_SCALE
        edge_attr_scaled = edge_attr / config.GLOBAL_SCALE

        # 3. GNN Processing
        feats = self.encoder(pos_centered, known_mask, side, batch_idx)
        feats_out, pos_out_centered = self.processor(feats, pos_centered, edge_index, edge_attr_scaled)
        
        # 4. Confidence Prediction (NEW)
        # This can be used during inference to flag uncertain predictions
        confidence = self.confidence_head(feats_out).view(B, N)  # (B, N)
        
        # 5. Residual with Confidence Weighting
        delta = pos_out_centered - pos_centered
        inv_mask = (~known_mask).float().unsqueeze(-1)
        
        # Apply confidence as soft gating (during training, all nodes learn)
        # During inference, low confidence flags potential issues
        delta_masked = delta.view(B, N, 3) * inv_mask.view(B, N, 1)
        
        # CRITICAL: Clip extreme deltas to prevent outliers
        delta_masked = torch.clamp(delta_masked, -0.5, 0.5)  # Limit movement
        
        pos_pred_centered = pos_centered.view(B, N, 3) + delta_masked
        
        # 6. Unscale
        pos_pred_final = (pos_pred_centered * config.GLOBAL_SCALE) + anchor_centroid
        
        return pos_pred_final.view(-1, 3), pos_init.view(-1, 3), confidence

    def predict_with_confidence(self, batch):
        """
        Inference method that returns predictions with confidence scores.
        Use this to identify potentially problematic predictions (like node 8).
        """
        pred, init, confidence = self.forward(batch)
        return pred, confidence