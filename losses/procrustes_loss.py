# losses/procrustes_loss.py
import torch


def procrustes_align(pred, target):
    B, N, _ = pred.shape

    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)

    H = torch.matmul(pred_c.transpose(1, 2), target_c)
    U, S, Vt = torch.linalg.svd(H)

    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    pred_aligned = torch.matmul(pred_c, R)
    pred_aligned += target.mean(dim=1, keepdim=True)

    return pred_aligned, R, S
