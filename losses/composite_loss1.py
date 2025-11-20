import torch
import torch.nn as nn
import config
from .procrustes_loss import procrustes_align
from .tps_loss import TPSWarp

class CompositeLoss1(nn.Module):
    def __init__(self, w_proc=1.0, w_icp=0.5, w_tps=1.0, w_edge=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        self.w_proc = w_proc
        self.w_icp = w_icp
        self.w_tps = w_tps
        self.w_edge = w_edge

        # Initialize TPS Warper with known IDs (anchors)
        # reg=0.01 is important if you only have 3 points!
        self.tps_warper = TPSWarp(control_indices=config.KNOWN_IDS, reg=0.01)

    def chamfer_distance(self, x, y):
        """
        ICP-style Loss: Measures distance to the NEAREST neighbor.
        Does not assume index correspondence. Good for structural sanity.
        """
        # x, y: (B, N, 3)
        dist = torch.cdist(x, y) # (B, N, N)
        
        # Min distance from Pred to any GT point
        min_dist_pred, _ = torch.min(dist, dim=2) 
        # Min distance from GT to any Pred point
        min_dist_target, _ = torch.min(dist, dim=1)
        
        return torch.mean(min_dist_pred) + torch.mean(min_dist_target)

    def forward(self, pred, target, known_mask, edge_index, edge_attr_gt):
        B, N, _ = pred.shape
        unknown = (~known_mask).float().unsqueeze(-1)

        # -------------------------------------------
        # 1. Procrustes Loss (Rigid Global Alignment)
        # -------------------------------------------
        # Aligns centroids and rotation optimally
        pred_rigid, _, _ = procrustes_align(pred, target)
        
        # Compare rigidly aligned prediction to target
        L_proc = self.mse(pred_rigid, target)

        # -------------------------------------------
        # 2. ICP Loss (Chamfer Distance)
        # -------------------------------------------
        # Mimics ICP by pulling points to their closest geometric neighbor 
        # rather than their index neighbor. Helps if prediction is jumbled.
        L_icp = self.chamfer_distance(pred_rigid, target)

        # -------------------------------------------
        # 3. TPS Loss (Non-Rigid Warping)
        # -------------------------------------------
        # Warp the rigid prediction so the Known Nodes match GT exactly.
        # Then check if the Unknown nodes fall in the right place.
        pred_warped = self.tps_warper(pred_rigid, target)
        
        # We care most about the unknown nodes here
        L_tps = self.l1(pred_warped * unknown, target * unknown)

        # -------------------------------------------
        # 4. Edge Consistency Loss
        # -------------------------------------------
        pred_flat = pred.view(-1, 3)
        row, col = edge_index
        pred_edge_vec = pred_flat[row] - pred_flat[col]
        pred_edge_dist = pred_edge_vec.norm(dim=-1, keepdim=True)
        
        L_edge = self.mse(pred_edge_dist, edge_attr_gt)

        # Total Weighted Loss
        total_loss = (self.w_proc * L_proc) + \
                     (self.w_icp * L_icp) + \
                     (self.w_tps * L_tps) + \
                     (self.w_edge * L_edge)

        return total_loss, {
            "L_proc": L_proc.item(), 
            "L_icp": L_icp.item(),
            "L_tps": L_tps.item(),
            "L_edge": L_edge.item()
        }