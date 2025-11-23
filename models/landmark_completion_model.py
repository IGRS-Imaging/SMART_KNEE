import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_mean

import config
from .encoder import Encoder
from .egnn_processor import EGNNProcessor
from .decoder import Decoder

class LandmarkCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # LOAD MEAN SHAPE
        try:
            # Load the canonical (e.g., Mean Right) shape
            mean_np = np.load(config.MEAN_SHAPE_PATH)
            self.register_buffer("mean_canonical_shape", torch.tensor(mean_np, dtype=torch.float32))
            print("Mean canonical shape loaded into model.")
        except:
            print(f"WARNING: Mean canonical shape not found at {config.MEAN_SHAPE_PATH}. Run utils/calculate_mean_shape.py first!")

        # Define the reflection matrix for Left/Right shapes (e.g., flipping the X-axis)
        # Assuming the canonical shape is Right. Left needs reflection.
        reflection_np = np.diag([-1, 1, 1]) 
        self.register_buffer("reflection_matrix", torch.tensor(reflection_np, dtype=torch.float32))

        self.encoder = Encoder(config.FEAT_DIM, config.HIDDEN_DIM, use_coords=True)
        
        self.processor = EGNNProcessor(
            feat_dim=config.FEAT_DIM,
            n_layers=config.GNN_LAYERS,
            edge_dim=config.EDGE_DIM, 
            m_dim=config.HIDDEN_DIM,
        )

        self.decoder = Decoder(config.FEAT_DIM, config.HIDDEN_DIM, 3)

    def batch_kabsch(self, pred_pts, target_pts, mask):
        """ 
        Rigid Anchoring (Kabsch Algorithm) with **Reflection Correction**. 
        Input shapes: (B, N, 3)
        """
        B, N, _ = pred_pts.shape
        device = pred_pts.device
        mask_float = mask.float().unsqueeze(-1)
        
        # Centroids (weighted by mask)
        count = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pred_c = (pred_pts * mask_float).sum(dim=1, keepdim=True) / count
        target_c = (target_pts * mask_float).sum(dim=1, keepdim=True) / count
        
        # Centered Points (weighted by mask)
        P = (pred_pts - pred_c) * mask_float
        Q = (target_pts - target_c) * mask_float
        
        # Cross-Covariance Matrix H
        H = torch.matmul(P.transpose(1, 2), Q)
        
        # SVD of H
        U, S, Vt = torch.linalg.svd(H)
        
        # --- REFLECTION FIX (R = U @ I @ V.T) ---
        # 1. Calculate determinant of the original R candidate (V @ U.T)
        R_candidate = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
        det = torch.det(R_candidate)
        
        # 2. Create the correction matrix 'I'
        I = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # 3. Apply flip mask to the last element of I for batches where det < 0
        flip_mask = (det < 0).long()
        I[:, 2, 2] = 1 - 2 * flip_mask # Sets to -1 if det < 0, else 1
        
        # 4. Final Rotation Matrix R = U @ I @ V.T (Forces det(R) = 1)
        R = torch.matmul(torch.matmul(U, I), Vt)
        # --- END OF REFLECTION FIX ---
        
        # Translation vector t = target_c - R @ pred_c
        t = target_c - torch.matmul(pred_c, R)
        return R, t

    def align_template(self, batch_size, pos_target, known_mask, side):
        """
        Aligns the Chirality-Specific Mean Shape Template to the current KNOWN nodes.
        Returns: pos_init (B, N, 3)
        """
        # 1. Prepare Canonical Template (B, N, 3)
        template = self.mean_canonical_shape.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 2. Apply Reflection if needed (for Left femurs, side=0)
        # Left femurs are denoted by side=0
        is_left = (side == 0)
        
        # If any are left, apply the reflection matrix
        if is_left.any():
            # Apply R_reflect to the template only for left femurs
            reflected_template = torch.matmul(template, self.reflection_matrix)
            template[is_left] = reflected_template[is_left]

        # 3. Calculate Transform mapping Chirality-Specific Template -> Target (using Known nodes only)
        R, t = self.batch_kabsch(template, pos_target, known_mask)
        
        # 4. Apply Transform
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

        # -------------------------------------------------------
        # 1. INITIALIZATION (CHIRALITY-AWARE TEMPLATE ALIGNMENT)
        # -------------------------------------------------------
        pos_init = self.align_template(B, pos_reshaped, known_mask_reshaped, side)
        
        # -------------------------------------------------------
        # 2. Normalize for GNN
        # -------------------------------------------------------
        centroid_init = pos_init.mean(dim=1, keepdim=True) # (B, 1, 3)
        
        # Flatten for scatter operations
        centroid_nodes = centroid_init.repeat(1, N, 1).view(-1, 3)
        
        pos_centered = (pos_init.view(-1, 3) - centroid_nodes) / config.GLOBAL_SCALE
        edge_attr_scaled = edge_attr / config.GLOBAL_SCALE

        # -------------------------------------------------------
        # 3. GNN Processing
        # -------------------------------------------------------
        feats = self.encoder(pos_centered, known_mask, side, batch_idx)
        
        # Predict DEFORMATION (Residuals)
        feats_out, pos_out = self.processor(feats, pos_centered, edge_index, edge_attr_scaled)
        delta = self.decoder(feats_out)
        
        # Apply deformation
        pos_pred_scaled = pos_out + delta

        # -------------------------------------------------------
        # 4. Unscale & Final Hard Anchor
        # -------------------------------------------------------
        pos_pred_unscaled = (pos_pred_scaled * config.GLOBAL_SCALE) + centroid_nodes
        
        # Final Hard Anchor (Kabsch with reflection fix) to ensure perfect alignment of known nodes
        pred_view = pos_pred_unscaled.view(B, N, 3)
        R_final, t_final = self.batch_kabsch(pred_view, pos_reshaped, known_mask_reshaped)
        
        pos_pred_final = torch.matmul(pred_view, R_final) + t_final
        
        return pos_pred_final.view(-1, 3), pos_init.view(-1, 3)