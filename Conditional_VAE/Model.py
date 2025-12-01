# import torch.nn as nn
# from Processor import Processor 
# from Encoder import Encoder 
# from Decoder import Decoder 
# from Encoder import reparam
# from Config import IN_DIM_COND, COND_EMB_DIM, LATENT_DIM, NUM_POINTS  

# class FemurVAE(nn.Module):
#     # ... rest of your class unchanged ...
#     """Full Conditional VAE for femur points."""
#     def __init__(self, in_dim_cond=IN_DIM_COND, cond_emb=COND_EMB_DIM, latent_dim=LATENT_DIM, num_points=NUM_POINTS):
#         super().__init__()
#         self.processor = Processor(in_dim_cond, cond_emb)
#         self.encoder = Encoder(3, cond_emb, latent_dim)
#         self.decoder = Decoder(latent_dim, cond_emb, num_points)

#     def forward(self, points, cond):
#         c = self.processor(cond)
#         mu, logvar = self.encoder(points, c)
#         z = reparam(mu, logvar)
#         recon = self.decoder(z, c)
#         return recon, mu, logvar

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FemurVAE(nn.Module):
#     def __init__(self, n_points=1024, latent_dim=256, cond_dim=6):  # Fixed: cond_dim=6
#         super(FemurVAE, self).__init__()
#         self.n_points = n_points
#         self.latent_dim = latent_dim
#         self.cond_dim = cond_dim
        
#         # Encoder input dim: flattened points + cond
#         encoder_input_dim = n_points * 3 + cond_dim  # 3072 + 6 = 3078
#         self.encoder = nn.Sequential(
#             nn.Linear(encoder_input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
#             nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
#             nn.Linear(256, latent_dim * 2)  # For mu + logvar
#         )
        
#         # Decoder input dim: z + cond
#         decoder_input_dim = latent_dim + cond_dim  # 256 + 6 = 262
#         self.decoder = nn.Sequential(
#             nn.Linear(decoder_input_dim, 1024), nn.ReLU(), nn.Dropout(0.1),  # Capacity
#             nn.Linear(1024, 512), nn.ReLU(),
#             nn.Linear(512, n_points * 3)  # Flat output
#         )

#     def encode(self, pts, cond):
#         batch_size = pts.size(0)
#         pts_flat = pts.view(batch_size, -1)  # (B, 3072)
#         h = torch.cat([pts_flat, cond], dim=1)  # (B, 3078)
#         out = self.encoder(h)
#         mu, logvar = out.chunk(2, dim=1)  # (B, 256) each
#         return mu, logvar

#     def reparametrize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z, cond):
#         batch_size = z.size(0)
#         h = torch.cat([z, cond], dim=1)  # (B, 262)
#         out = self.decoder(h)  # (B, 3072)
#         recon = out.view(batch_size, self.n_points, 3)  # (B, 1024, 3)
#         return recon

#     def forward(self, pts, cond):
#         mu, logvar = self.encode(pts, cond)
#         z = self.reparametrize(mu, logvar)
#         recon = self.decode(z, cond)
#         return recon, mu, logvar



import torch
import torch.nn as nn
import torch.nn.functional as F

class FemurVAE(nn.Module):
    def __init__(self, n_points=1024, latent_dim=256, cond_dim=42):  # FIXED: 42
        super(FemurVAE, self).__init__()
        self.n_points = n_points
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim  
        # Encoder
        encoder_input_dim = n_points * 3 + cond_dim  # 3072 + 42 = 3114
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim * 2)
        )
        
        # Decoder
        decoder_input_dim = latent_dim + cond_dim  # 256 + 42 = 298
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, n_points * 3)
        )

    def encode(self, pts, cond):
        batch_size = pts.size(0)
        pts_flat = pts.view(batch_size, -1)
        h = torch.cat([pts_flat, cond], dim=1)
        out = self.encoder(h)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        batch_size = z.size(0)
        h = torch.cat([z, cond], dim=1)
        out = self.decoder(h)
        recon = out.view(batch_size, self.n_points, 3)
        return recon

    def forward(self, pts, cond):
        mu, logvar = self.encode(pts, cond)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar