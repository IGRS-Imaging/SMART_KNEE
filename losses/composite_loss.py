import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_edge=1.0, w_angle=1.0, w_global=1.0):
        super().__init__()
        self.base_criterion = nn.L1Loss(reduction='none') 
        self.edge_criterion = nn.L1Loss()
        self.angle_criterion = nn.CosineEmbeddingLoss()
        self.global_criterion = nn.L1Loss()
        
        self.w_pos = w_pos
        self.w_edge = w_edge
        self.w_angle = w_angle
        self.w_global = w_global

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt):
        # 1. Masked Position Loss (Hit the targets)
        per_node_loss = self.base_criterion(pred, target).mean(dim=-1)
        # Weights: 1.0 for Known, 1.0 for Unknown (Uniform is usually stable with Anchors)
        # We rely on Hard Anchors to handle the "Known" constraint naturally.
        L_pos = per_node_loss.mean()

        # 2. Local Edge Length Loss (Defined connections)
        pred_flat = pred.view(-1, 3)
        target_flat = target.view(-1, 3)
        row, col = edge_index
        
        pred_vec = pred_flat[row] - pred_flat[col]
        target_vec = target_flat[row] - target_flat[col]
        
        pred_dist = pred_vec.norm(dim=-1, keepdim=True)
        L_edge = self.edge_criterion(pred_dist, edge_attr_gt)

        # 3. Orientation Loss (Prevent Folding)
        target_labels = torch.ones(row.size(0), device=pred.device)
        L_angle = self.angle_criterion(pred_vec, target_vec, target_labels)

        # 4. Global Structure Loss (New: All-pairs distance preservation)
        # This prevents the shape from "crumpling" in areas without explicit edges.
        B, N, _ = pred.shape
        # Compute Pairwise Distances for Pred
        # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
        diff_pred = pred.unsqueeze(2) - pred.unsqueeze(1)
        dist_pred = torch.norm(diff_pred, dim=-1) # (B, N, N)
        
        # Compute Pairwise Distances for Target
        diff_target = target.unsqueeze(2) - target.unsqueeze(1)
        dist_target = torch.norm(diff_target, dim=-1) # (B, N, N)
        
        # Minimize difference in the "internal shape matrix"
        L_global = self.global_criterion(dist_pred, dist_target)

        # Weighted Sum
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