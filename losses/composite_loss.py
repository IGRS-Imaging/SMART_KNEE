# losses/composite_loss.py

import torch
import torch.nn as nn
import config
from .procrustes_loss import procrustes_align


def pairwise_distances(x):
    """
    x: (B, N, 3)
    Returns D: (B, N, N) with Euclidean distances.
    """
    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,3)
    D = torch.linalg.norm(diff, dim=-1)     # (B,N,N)
    return D


class CompositeLoss(nn.Module):
    """
    L = L_align + lambda_shape * L_shape
      - L_align: MSE after Procrustes alignment.
      - L_shape: L1 between pairwise distance matrices.
    """

    def __init__(self, lambda_shape: float = None):
        super().__init__()
        self.lambda_shape = lambda_shape if lambda_shape is not None else config.LAMBDA_SHAPE
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        """
        pred:   (B,N,3)
        target: (B,N,3)
        """
        # 1) Procrustes alignment
        pred_aligned, _, _ = procrustes_align(pred, target)
        L_align = self.mse(pred_aligned, target)

        # 2) Pairwise distance loss
        D_pred = pairwise_distances(pred_aligned)
        D_tgt = pairwise_distances(target)
        L_shape = self.l1(D_pred, D_tgt)

        loss = L_align + self.lambda_shape * L_shape

        info = {
            "L_total": loss.item(),
            "L_align": L_align.item(),
            "L_shape": L_shape.item(),
        }
        return loss, info
