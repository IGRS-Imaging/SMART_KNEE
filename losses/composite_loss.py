# losses/composite_loss.py
import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_edge=0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss()
        
        self.w_pos = w_pos
        self.w_edge = w_edge 

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt):
        """
        pred: (B, N, 3) - Already rigidly anchored by the model
        target: (B, N, 3) - Absolute Ground Truth
        """
        B, N, _ = pred.shape
        
        # 1. Position Loss (Direct MSE)
        # We calculate MSE on ALL nodes.
        # Since the model anchors the known nodes internally, 
        # the loss on known nodes should naturally go to near-zero.
        # The loss on unknown nodes drives the learning of the shape.
        pos_loss_per_node = self.mse(pred, target).sum(dim=-1) # (B, N)
        
        # You can optionally weight unknown nodes higher if needed, 
        # but standard MSE is usually sufficient with hard anchoring.
        L_pos = pos_loss_per_node.mean()

        # 2. Edge Consistency Loss
        # Enforces physical structure (bone length consistency)
        pred_flat = pred.view(-1, 3)
        row, col = edge_index
        
        pred_vec = pred_flat[row] - pred_flat[col]
        pred_dist = pred_vec.norm(dim=-1, keepdim=True) # (E, 1)
        
        # Compare predicted edge lengths to Ground Truth edge lengths
        L_edge = nn.MSELoss()(pred_dist, edge_attr_gt)

        # Total Loss
        total_loss = (self.w_pos * L_pos) + (self.w_edge * L_edge)

        return total_loss, {
            "L_pos": L_pos.item(), 
            "L_edge": L_edge.item(),
            "L_align": 0.0 # Deprecated placeholder
        }