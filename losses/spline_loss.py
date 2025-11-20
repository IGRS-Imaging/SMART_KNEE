# losses/spline_loss.py
import torch
import torch.nn as nn

class CubicSplineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred):
        """
        Enforces smoothness by minimizing the second derivative (curvature).
        Approximates the energy function of a Cubic Spline.
        
        pred: (B, N, 3)
        """
        # 1. Sort points by the Z-axis (longitudinal axis of the bone)
        # This automatically orders the landmarks from Hip -> Knee
        # regardless of their ID, ensuring we trace a line down the bone.
        # (Change '2' to '1' or '0' if your bone is oriented along Y or X)
        sorted_vals, sort_indices = torch.sort(pred[:, :, 2], dim=1)
        
        # Gather the full 3D coordinates in this sorted order
        batch_indices = torch.arange(pred.size(0)).unsqueeze(1).expand_as(sort_indices).to(pred.device)
        pred_sorted = pred[batch_indices, sort_indices] # (B, N, 3) ordered by height

        # 2. Calculate First Difference (Velocity / Tangent)
        # P_i+1 - P_i
        diff1 = pred_sorted[:, 1:, :] - pred_sorted[:, :-1, :]

        # 3. Calculate Second Difference (Acceleration / Curvature)
        # (P_i+2 - P_i+1) - (P_i+1 - P_i)  =  P_i+2 - 2*P_i+1 + P_i
        diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]

        # 4. Minimize the magnitude of the curvature
        # This forces the points to form a straight-ish, smooth line (Cubic Spline behavior)
        loss = diff2.norm(dim=-1).mean()
        
        return loss