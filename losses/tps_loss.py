import torch
import torch.nn as nn

class TPSLoss(nn.Module):
    def __init__(self, mean_shape, alpha=0.01):
        super().__init__()
        self.mean_shape = mean_shape
        self.alpha = alpha
        
        # Precompute Bending Energy Matrix (L_tps)
        with torch.no_grad():
            N, _ = mean_shape.shape
            device = mean_shape.device
            
            # 1. Construct K (N, N) - Kernel Matrix
            # U(r) = -r for 3D TPS
            diff = mean_shape.unsqueeze(0) - mean_shape.unsqueeze(1)
            dist = torch.norm(diff, dim=-1)
            K = -dist 
            
            # 2. Construct P (N, 4) - Affine Basis [1, x, y, z]
            ones = torch.ones(N, 1, device=device)
            P = torch.cat([ones, mean_shape], dim=1) # (N, 4)
            
            # 3. Construct Bordered Matrix M (N+4, N+4)
            # | K   P |
            # | P^T 0 |
            M_top = torch.cat([K, P], dim=1)      # (N, N+4)
            M_bot = torch.cat([P.T, torch.zeros(4, 4, device=device)], dim=1) # (4, N+4)
            M = torch.cat([M_top, M_bot], dim=0)  # (N+4, N+4)
            
            # 4. Invert M
            # Add slight jitter for numerical stability
            M_inv = torch.inverse(M + torch.eye(N+4, device=device) * 1e-6)
            
            # 5. Extract top-left N x N block (L_tps)
            # This matrix is the Bending Energy Matrix and is Positive Semi-Definite.
            self.L_tps = M_inv[:N, :N]

    def forward(self, pred):
        """
        Calculates the Bending Energy of the deformation from Mean Shape -> Pred.
        pred: (B, N, 3)
        """
        # Ensure L_tps is on the correct device
        if self.L_tps.device != pred.device:
            self.L_tps = self.L_tps.to(pred.device)
            
        # Energy = Trace(Y^T * L_tps * Y)
        # Dimensions: (B, 3, N) @ (N, N) @ (B, N, 3)
        pred_t = pred.transpose(1, 2)
        
        # 1. Multiply by Bending Matrix
        term1 = torch.matmul(pred_t, self.L_tps)
        
        # 2. Multiply by Coordinates
        energy_matrix = torch.matmul(term1, pred)
        
        # 3. Trace (Sum of diagonal elements)
        bending_energy = energy_matrix.diagonal(dim1=-2, dim2=-1).sum(-1) # (B,)
        
        # Return mean energy scaled by alpha
        return self.alpha * bending_energy.mean()