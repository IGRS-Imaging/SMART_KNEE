import torch
import torch.nn as nn
from .procrustes_loss import procrustes_align

class CompositeLoss(nn.Module):
    def __init__(self, w_align=1.0, w_shape=0.1, w_edge=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        self.w_align = w_align
        self.w_shape = w_shape
        self.w_edge = w_edge  # Weight for the edge constraint

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt):
        """
        pred: (B, N, 3)
        target: (B, N, 3)
        edge_index: (2, E_total)
        edge_attr_gt: (E_total, 1)
        """
        B, N, _ = pred.shape
        unknown = (~known_mask).float().unsqueeze(-1)

        # 1. Alignment Loss (Procrustes)
        pred_aligned, _, _ = procrustes_align(pred, target)
        
        # Calculate error only on unknown nodes? Or all?
        # Usually better to supervise all to keep structure, but heavily weight unknown
        L_align = self.mse(pred_aligned * unknown, target * unknown)

        # 2. Global Shape Loss (Pairwise distances matrix)
        # Downsamples if N is large, but for 12 nodes this is fine
        pd = torch.cdist(pred, pred)
        td = torch.cdist(target, target)
        L_shape = self.l1(pd, td)

        # 3. Edge Consistency Loss (Explicit Constraint)
        # We must extract edges from the batch-wise prediction
        # Reshape pred to (B*N, 3) to use with edge_index
        pred_flat = pred.view(-1, 3)
        
        row, col = edge_index
        pred_edge_vec = pred_flat[row] - pred_flat[col]
        pred_edge_dist = pred_edge_vec.norm(dim=-1, keepdim=True) # (E, 1)
        
        # Compare predicted edge lengths to Ground Truth edge lengths from CSV
        L_edge = self.mse(pred_edge_dist, edge_attr_gt)

        total_loss = (self.w_align * L_align) + \
                     (self.w_shape * L_shape) + \
                     (self.w_edge * L_edge)

        return total_loss, {
            "L_align": L_align.item(), 
            "L_shape": L_shape.item(),
            "L_edge": L_edge.item()
        }