import torch
import torch.nn as nn

class TPSWarp(nn.Module):
    def __init__(self, control_indices, reg=1e-3):
        """
        Args:
            control_indices: List of indices for known landmarks (anchors).
            reg: Regularization to prevent singular matrices (crucial for <4 points).
        """
        super().__init__()
        self.control_indices = control_indices
        self.reg = reg

    def tps_kernel(self, dist):
        """3D TPS Kernel U(r) = -r"""
        return -dist

    def forward(self, src, tgt):
        """
        Warps 'src' to match 'tgt' based on control points.
        Returns the warped source shape.
        """
        B, N, _ = src.shape
        device = src.device
        
        # 1. Extract Control Points
        # We assume src and tgt are already somewhat aligned (Procrustes)
        # control_src: (B, K, 3), control_tgt: (B, K, 3)
        control_src = src[:, self.control_indices, :]
        control_tgt = tgt[:, self.control_indices, :]
        K = control_src.shape[1]

        # 2. Construct System Matrix L
        # K_mat: Distance between source control points
        dist_kk = torch.cdist(control_src, control_src)
        K_mat = self.tps_kernel(dist_kk)
        
        # Regularization (Identity * reg)
        K_mat = K_mat + torch.eye(K, device=device).unsqueeze(0) * self.reg
        
        # P_mat: [1, x, y, z]
        ones = torch.ones(B, K, 1, device=device)
        P_mat = torch.cat([ones, control_src], dim=-1) # (B, K, 4)

        # Assemble L: [[K, P], [P^T, 0]]
        L_top = torch.cat([K_mat, P_mat], dim=-1)
        L_bot = torch.cat([P_mat.transpose(1, 2), torch.zeros(B, 4, 4, device=device)], dim=-1)
        L = torch.cat([L_top, L_bot], dim=1) # (B, K+4, K+4)

        # 3. Target Vector Y
        # Y = [[Target Control Coords], [0]]
        zeros = torch.zeros(B, 4, 3, device=device)
        Y = torch.cat([control_tgt, zeros], dim=1) # (B, K+4, 3)

        # 4. Solve for Weights (WA)
        # L * WA = Y  => WA = L^-1 * Y
        # Using least squares (lstsq) is safer for stability than solve
        try:
            WA = torch.linalg.solve(L, Y)
        except RuntimeError:
            WA = torch.linalg.lstsq(L, Y).solution
            
        # 5. Apply Warping to ALL points in 'src'
        W = WA[:, :K, :] # Weights
        A = WA[:, K:, :] # Affine parameters

        # Affine part
        ones_N = torch.ones(B, N, 1, device=device)
        P_full = torch.cat([ones_N, src], dim=-1)
        affine_term = torch.matmul(P_full, A)

        # Non-rigid part
        dist_nk = torch.cdist(src, control_src)
        kernel_nk = self.tps_kernel(dist_nk)
        warp_term = torch.matmul(kernel_nk, W)

        src_warped = affine_term + warp_term
        return src_warped