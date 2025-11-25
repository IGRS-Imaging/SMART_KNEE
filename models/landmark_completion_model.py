#model/landmark_completion_model.py
import torch
import torch.nn as nn
import numpy as np
import config
from .encoder import Encoder
from .egnn_processor import EGNNProcessor
from .decoder import Decoder

class LandmarkCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Learnable Template Initialization
        try:
            mean_np = np.load(config.MEAN_SHAPE_PATH)
            # Normalize the loaded shape to match network scale immediately
            mean_tensor = torch.tensor(mean_np, dtype=torch.float32)
            self.mean_canonical_shape = nn.Parameter(mean_tensor)
            print("✅ Initialized Learnable Template from file.")
        except:
            print("⚠️ Warning: Mean shape file not found. Initializing random.")
            self.mean_canonical_shape = nn.Parameter(torch.randn(config.NUM_NODES, 3))

        # Reflection matrix (Fixed buffer)
        self.register_buffer("reflection_matrix", torch.tensor(np.diag([-1, 1, 1]), dtype=torch.float32))

        self.encoder = Encoder(config.FEAT_DIM, config.HIDDEN_DIM, use_coords=True)
        
        self.processor = EGNNProcessor(
            feat_dim=config.FEAT_DIM,
            n_layers=config.GNN_LAYERS,
            edge_dim=config.EDGE_DIM, 
            m_dim=config.HIDDEN_DIM,
        )

        self.decoder = Decoder(config.FEAT_DIM, config.HIDDEN_DIM, 3)

    def batch_kabsch(self, pred_pts, target_pts, mask):
        B, N, _ = pred_pts.shape
        device = pred_pts.device
        mask_float = mask.float().unsqueeze(-1)
        
        # Robust count to avoid div by zero
        count = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
        
        pred_c = (pred_pts * mask_float).sum(dim=1, keepdim=True) / count
        target_c = (target_pts * mask_float).sum(dim=1, keepdim=True) / count
        
        P = (pred_pts - pred_c) * mask_float
        Q = (target_pts - target_c) * mask_float
        
        H = torch.matmul(P.transpose(1, 2), Q)
        U, S, Vt = torch.linalg.svd(H)
 #################################################################################3       
        # Reflection Correction
        R_candidate = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
        det = torch.det(R_candidate)
        
        I = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        flip_mask = (det < 0).long()
        I[:, 2, 2] = 1 - 2 * flip_mask 
        
        R = torch.matmul(torch.matmul(U, I), Vt)
        t = target_c - torch.matmul(pred_c, R)
        return R, t
############################################################################
    def align_template(self, batch_size, pos_target, known_mask, side):
        # Expand template
        template = self.mean_canonical_shape.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Handle Left/Right reflection
        is_left = (side == 0)
        if is_left.any():
            reflected_template = torch.matmul(template, self.reflection_matrix)
            template[is_left] = reflected_template[is_left]

        # Detach template for alignment calculation (prevents gradients flowing into rotation)
        template_detached = template.detach()
        
        R, t = self.batch_kabsch(template_detached, pos_target, known_mask)
        aligned_template = torch.matmul(template, R) + t
        
        return aligned_template

    def forward(self, batch):
        pos = batch.pos
        known_mask = batch.known_mask
        side = batch.side
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        B = side.size(0)
        N = config.NUM_NODES
        pos_reshaped = pos.view(B, N, 3)
        known_mask_reshaped = known_mask.view(B, N)

        # 1. Initialization
        pos_init = self.align_template(B, pos_reshaped, known_mask_reshaped, side)
        
        # 2. Normalize
        centroid_init = pos_init.mean(dim=1, keepdim=True)
        centroid_nodes = centroid_init.repeat(1, N, 1).view(-1, 3)
        
        # Normalize input
        pos_centered = (pos_init.view(-1, 3) - centroid_nodes) / config.GLOBAL_SCALE
        edge_attr_scaled = edge_attr / config.GLOBAL_SCALE

        # 3. GNN Processing
        feats = self.encoder(pos_centered, known_mask, side, batch_idx)
        feats_out, pos_out = self.processor(feats, pos_centered, edge_index, edge_attr_scaled)
        delta = self.decoder(feats_out)
        ############################################################################
        # 4. Unscale
        pos_pred_scaled = pos_out + delta
        pos_pred_unscaled = (pos_pred_scaled * config.GLOBAL_SCALE) + centroid_nodes
        ############################################################################
        # 5. Final Hard Anchor
        pred_view = pos_pred_unscaled.view(B, N, 3)
        R_final, t_final = self.batch_kabsch(pred_view, pos_reshaped, known_mask_reshaped)
        
        pos_pred_final = torch.matmul(pred_view, R_final) + t_final
        
        return pos_pred_final.view(-1, 3), pos_init.view(-1, 3)