import torch
import torch.nn as nn
import config

class CompositeLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_edge=1.0, w_angle=1.0, w_global=1.0):
        super().__init__()
        # SmoothL1Loss (Stable Precision)
        self.base_criterion = nn.SmoothL1Loss(reduction='none', beta=1.0) 
        
        self.edge_criterion = nn.L1Loss()
        self.angle_criterion = nn.CosineEmbeddingLoss()
        self.global_criterion = nn.L1Loss()
        
        self.w_pos = w_pos
        self.w_edge = w_edge
        self.w_angle = w_angle
        self.w_global = w_global

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt):
        # 1. Masked Position Loss
        per_node_loss = self.base_criterion(pred, target).mean(dim=-1) # (B, N)
        
        # --- DYNAMIC HARD EXAMPLE MINING ---
        with torch.no_grad():
            raw_error = torch.norm(pred - target, dim=-1)
            mean_error = raw_error.mean(dim=1, keepdim=True).clamp_min(1e-6)
            
            # Dynamic Weights: Focus on nodes with higher-than-average error
            # Clipped to 3.0 to keep training stable
            node_weights = (raw_error / mean_error).clamp(min=1.0, max=3.0)

        # Weighted Mean
        L_pos = (per_node_loss * node_weights).mean()

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

        # 4. Global Structure Loss
        diff_pred = pred.unsqueeze(2) - pred.unsqueeze(1)
        dist_pred = torch.norm(diff_pred, dim=-1)
        
        diff_target = target.unsqueeze(2) - target.unsqueeze(1)
        dist_target = torch.norm(diff_target, dim=-1)
        
        L_global = self.global_criterion(dist_pred, dist_target)

        total_loss = (self.w_pos * L_pos) + \
                     (self.w_edge * L_edge) + \
                     (self.w_angle * L_angle) + \
                     (self.w_global * L_global)

        return total_loss, {
            "L_pos": L_pos.item(), 
            "L_edge": L_edge.item(),
            "L_angle": L_angle.item(),
            "L_global": L_global.item()
        }