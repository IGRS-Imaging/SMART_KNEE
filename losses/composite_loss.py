# losses/composite_loss.py
import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_edge=0.5):
        super().__init__()
        # CHANGED TO L1LOSS (MAE) - Better for geometric convergence
        self.pos_criterion = nn.L1Loss() 
        self.edge_criterion = nn.L1Loss()
        
        self.w_pos = w_pos
        self.w_edge = w_edge 

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt):
        """
        pred: (B, N, 3)
        target: (B, N, 3)
        """
        # 1. Position Loss (L1)
        # We want to minimize distance for ALL nodes.
        L_pos = self.pos_criterion(pred, target)

        # 2. Edge Consistency Loss (L1)
        pred_flat = pred.view(-1, 3)
        row, col = edge_index
        
        pred_vec = pred_flat[row] - pred_flat[col]
        pred_dist = pred_vec.norm(dim=-1, keepdim=True) # (E, 1)
        
        L_edge = self.edge_criterion(pred_dist, edge_attr_gt)

        total_loss = (self.w_pos * L_pos) + (self.w_edge * L_edge)

        return total_loss, {
            "L_pos": L_pos.item(), 
            "L_edge": L_edge.item()
        }