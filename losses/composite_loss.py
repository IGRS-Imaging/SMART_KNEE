import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, w_pos=10.0, w_edge=2.0, w_angle=1.0):
        super().__init__()
        # SmoothL1 is robust against exploding gradients (acts like L1 for large errors)
        self.pos_criterion = nn.SmoothL1Loss(beta=10.0) 
        self.edge_criterion = nn.L1Loss()
        self.angle_criterion = nn.CosineEmbeddingLoss()
        
        self.w_pos = w_pos
        self.w_edge = w_edge
        self.w_angle = w_angle

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt):
        # 1. Position Loss
        L_pos = self.pos_criterion(pred, target)

        # 2. Edge Length Loss
        pred_flat = pred.view(-1, 3)
        target_flat = target.view(-1, 3)
        row, col = edge_index
        
        pred_vec = pred_flat[row] - pred_flat[col]
        target_vec = target_flat[row] - target_flat[col]
        
        pred_dist = pred_vec.norm(dim=-1, keepdim=True)
        L_edge = self.edge_criterion(pred_dist, edge_attr_gt)

        # 3. Orientation Loss
        target_labels = torch.ones(row.size(0), device=pred.device)
        L_angle = self.angle_criterion(pred_vec, target_vec, target_labels)

        total_loss = (self.w_pos * L_pos) + (self.w_edge * L_edge) + (self.w_angle * L_angle)

        return total_loss, {
            "L_pos": L_pos.item(), 
            "L_edge": L_edge.item(),
            "L_angle": L_angle.item()
        }