# losses/composite_loss.py
import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_edge=1.0, w_angle=1.0, w_global=1.0):
        super().__init__()
        # Use simple L1 for position to avoid squaring large errors (outliers)
        self.base_criterion = nn.L1Loss(reduction='none') 
        self.edge_criterion = nn.L1Loss(reduction='none')
        self.angle_criterion = nn.CosineEmbeddingLoss(reduction='mean')
        self.global_criterion = nn.L1Loss()
        
        self.w_pos = w_pos
        self.w_edge = w_edge
        self.w_angle = w_angle
        self.w_global = w_global

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt, epoch=None, max_epochs=None):
        """
        pred, target: (B, N, 3)
        known_mask:   (B, N)
        """
        
        # 1. POSITION LOSS + ROBUST OHEM
        # We perform Hard Example Mining, but we DO NOT anneal it down.
        # We keep the pressure on the hard nodes throughout training.
        per_node_loss = self.base_criterion(pred, target).mean(dim=-1) # (B, N)
        
        with torch.no_grad():
            raw_error = torch.norm(pred - target, dim=-1)
            mean_error = raw_error.mean(dim=1, keepdim=True).clamp_min(1e-6)
            # Cap the weight at 3.0 to prevent instability, but don't reduce it later
            node_weights = (raw_error / mean_error).clamp(1.0, 3.0)

        L_pos = (per_node_loss * node_weights).mean()

        # 2. EDGE LOSS (UNIFORM WEIGHTING)
        # CRITICAL FIX: Removed 1.0/dist weighting. 
        # Long edges (structural) are now just as important as short ones.
        B, N, _ = pred.shape
        pred_flat = pred.view(-1, 3)
        row, col = edge_index
        
        pred_vec = pred_flat[row] - pred_flat[col]
        pred_dist = torch.norm(pred_vec, dim=-1, keepdim=True)
        
        # Standard L1 loss on edge lengths
        L_edge = self.edge_criterion(pred_dist, edge_attr_gt).mean()

        # 3. ANGLE LOSS
        target_flat = target.view(-1, 3)
        target_vec = target_flat[row] - target_flat[col]
        target_labels = torch.ones(row.size(0), device=pred.device)
        L_angle = self.angle_criterion(pred_vec, target_vec, target_labels)

        # 4. GLOBAL STRUCTURE LOSS (CONSTANT)
        # CRITICAL FIX: Removed annealing. Global shape must hold firm at the end.
        diff_pred = pred.unsqueeze(2) - pred.unsqueeze(1) # (B, N, N, 3)
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