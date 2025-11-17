# losses/composite_loss.py
import torch
import torch.nn as nn
from .procrustes_loss import procrustes_align


class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, known_mask):
        B, N, _ = pred.shape

        unknown = (~known_mask).float().unsqueeze(-1)

        pred_u = pred * unknown
        target_u = target * unknown

        # alignment loss
        pred_aligned, _, _ = procrustes_align(pred, target)
        pred_aligned_u = pred_aligned * unknown

        L_align = self.mse(pred_aligned_u, target_u)

        # shape loss
        pd = torch.cdist(pred, pred)
        td = torch.cdist(target, target)
        L_shape = torch.mean(torch.abs(pd - td))

        return L_align + 0.1 * L_shape, {"L_align": L_align.item(), "L_shape": L_shape.item()}
