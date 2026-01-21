# losses/composite_loss.py
import torch
import torch.nn as nn
import config  # Import config to access NODE_IMPORTANCE

class CompositeLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_edge=1.0, w_angle=1.0, w_local_struct=1.0):
        super().__init__()
        self.base_criterion = nn.L1Loss(reduction='none')
        self.edge_criterion = nn.L1Loss(reduction='none')
        self.angle_criterion = nn.CosineEmbeddingLoss(reduction='mean')
        
        self.w_pos = w_pos
        self.w_edge = w_edge
        self.w_angle = w_angle
        self.w_local_struct = w_local_struct

        # --- NEW: Pre-load Manual Weights ---
        # Initialize all weights to 1.0 (Global Weight)
        self.manual_node_weights = torch.ones(config.NUM_NODES)
        
        # Apply specific weights from config
        if hasattr(config, 'NODE_IMPORTANCE') and config.NODE_IMPORTANCE:
            print(f"✅ Loading Manual Node Weights: {config.NODE_IMPORTANCE}")
            for node_idx, weight in config.NODE_IMPORTANCE.items():
                if 0 <= node_idx < config.NUM_NODES:
                    self.manual_node_weights[node_idx] = weight
        # ------------------------------------

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt, epoch=None, max_epochs=None):
        """
        pred, target: (B, N, 3)
        """
        device = pred.device
        
        # 1. POSITION LOSS
        per_node_loss = self.base_criterion(pred, target).mean(dim=-1)  # (B, N)
        
        # A. Dynamic weighting (keeps your existing logic for general difficulty)
        with torch.no_grad():
            raw_error = torch.norm(pred - target, dim=-1)
            batch_mean = raw_error.mean(dim=0, keepdim=True)  # (1, N)
            # Normalize so average weight is 1.0, cap at 2.5
            dynamic_weights = (batch_mean / (batch_mean.mean() + 1e-6)).clamp(0.5, 2.5)
            dynamic_weights = dynamic_weights.expand_as(per_node_loss)

        # B. Manual weighting (NEW)
        # Move manual weights to the correct device
        manual_weights = self.manual_node_weights.to(device).unsqueeze(0) # (1, N)
        
        # Combine: Final Weight = Dynamic * Manual
        # Example: If Node 11 is hard (dynamic=2.0) AND important (manual=3.0), 
        # total weight = 6.0. This forces the model to fix it.
        combined_weights = dynamic_weights * manual_weights

        # Calculate weighted Position Loss
        L_pos = (per_node_loss * combined_weights).mean()

        # 2. EDGE LOSS (Existing logic)
        B, N, _ = pred.shape
        pred_flat = pred.view(-1, 3)
        row, col = edge_index
        
        pred_vec = pred_flat[row] - pred_flat[col]
        pred_dist = torch.norm(pred_vec, dim=-1, keepdim=True)
        
        edge_weights = 1.0 / (edge_attr_gt + 1.0)
        edge_weights = edge_weights / (edge_weights.mean() + 1e-6)
        edge_weights = edge_weights.clamp(0.5, 2.0)
        
        edge_loss_per_edge = self.edge_criterion(pred_dist, edge_attr_gt).squeeze()
        L_edge = (edge_loss_per_edge * edge_weights.squeeze()).mean()

        # 3. ANGLE LOSS (Existing logic)
        target_flat = target.view(-1, 3)
        target_vec = target_flat[row] - target_flat[col]
        target_labels = torch.ones(row.size(0), device=device)
        L_angle = self.angle_criterion(pred_vec, target_vec, target_labels)

        # 4. LOCAL STRUCTURE (Existing logic)
        pred_dist_1hop = torch.norm(pred_flat[row] - pred_flat[col], dim=-1)
        target_dist_1hop = torch.norm(target_flat[row] - target_flat[col], dim=-1)
        L_local_struct = nn.functional.l1_loss(pred_dist_1hop, target_dist_1hop)

        total_loss = (self.w_pos * L_pos) + \
                     (self.w_edge * L_edge) + \
                     (self.w_angle * L_angle) + \
                     (self.w_local_struct * L_local_struct)

        return total_loss, {
            "L_pos": L_pos.item(), 
            "L_edge": L_edge.item(),
            "L_angle": L_angle.item(),
            "L_local_struct": L_local_struct.item()
        }