# losses/procrustes_loss.py

import torch


def procrustes_align(pred, gt):
    """
    Differentiable Procrustes alignment.
    pred: (B, N, 3)
    gt:   (B, N, 3)

    Returns:
        pred_aligned: (B, N, 3)
        R: (B, 3, 3)
        t: (B, 1, 3)
    """
    B, N, _ = pred.shape

    mu_pred = pred.mean(dim=1, keepdim=True)
    mu_gt = gt.mean(dim=1, keepdim=True)

    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    C = torch.matmul(gt_c.transpose(1, 2), pred_c) / N  # (B,3,3)

    U, S, Vt = torch.linalg.svd(C, full_matrices=False)

    # initial rotation
    R = torch.matmul(U, Vt)

    # Fix reflections: det(R) should be +1
    det_R = torch.det(R)  # (B,)
    D = torch.ones((B, 3, 3), device=pred.device)
    D[:, 2, 2] = torch.sign(det_R)
    R = torch.matmul(torch.matmul(U, D), Vt)

    t = mu_gt - torch.matmul(mu_pred, R)  # (B,1,3)

    pred_aligned = torch.matmul(pred, R) + t

    return pred_aligned, R, t
