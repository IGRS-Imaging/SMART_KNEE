# losses/procrustes_loss.py
import torch

def procrustes_align(pred, target, return_transform=False):
    """
    Aligns 'pred' to 'target' using Singular Value Decomposition (SVD).
    Ensures NO reflection (determinant of R is forced to +1).
    """
    B, N, _ = pred.shape

    # 1. Center both sets of points
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    
    pred_c = pred - pred_mean
    target_c = target - target_mean

    # 2. Compute Covariance Matrix H
    H = torch.matmul(pred_c.transpose(1, 2), target_c)

    # 3. SVD
    U, S, Vt = torch.linalg.svd(H)

    # 4. Compute Rotation Matrix R
    # V * U^T
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # 5. Reflection Correction (Determinant check)
    # If det(R) < 0, we have a reflection. We must flip the last column of V.
    det = torch.det(R)
    
    # Create a correction matrix diag(1, 1, det)
    I = torch.eye(3, device=pred.device).unsqueeze(0).repeat(B, 1, 1)
    I[:, 2, 2] = det
    
    # Recalculate R with correction
    # R = V * I * U^T
    R = torch.matmul(torch.matmul(Vt.transpose(1, 2), I), U.transpose(1, 2))

    # 6. Apply Transformation
    # pred_aligned = (pred - mu_p) * R + mu_t
    pred_aligned = torch.matmul(pred_c, R) + target_mean

    if return_transform:
        return pred_aligned, R, target_mean - torch.matmul(pred_mean, R)
        
    return pred_aligned, R, S