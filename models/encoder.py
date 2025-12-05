# models/encoder.py
import torch
import torch.nn as nn
import config

class Encoder(nn.Module):
    def __init__(self, feat_dim, hidden, use_coords=True):
        super().__init__()

        self.use_coords = use_coords

        # Embeddings
        self.chirality_emb = nn.Embedding(2, hidden)              # Left / Right
        self.node_index_emb = nn.Embedding(config.NUM_NODES, hidden)  # node id
        self.known_emb = nn.Embedding(2, hidden)                  # known=1, unknown=0

        coord_dim = 3 if use_coords else 0
        
        # TRIANGULATION FEATURES:
        # We add 3 distances (distance to each of the 3 anchors)
        # + 1 distance to centroid
        geo_dim = 4 
        
        in_dim = hidden * 3 + coord_dim + geo_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), # Added extra layer for depth
            nn.SiLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, pos, known_mask, side, batch_idx):
        """
        pos: (N_total, 3) - These are the Initial (Template-Aligned) positions
        """
        device = pos.device
        N_total = pos.size(0)
        
        # 1. Basic Embeddings
        side_embed = self.chirality_emb(side)[batch_idx]     # (N, hidden)
        local_idx = torch.arange(N_total, device=device) % config.NUM_NODES
        node_embed = self.node_index_emb(local_idx)          # (N, hidden)
        known_embed = self.known_emb(known_mask.long())      # (N, hidden)

        feats = [side_embed, node_embed, known_embed]

        # 2. Coordinate Features
        if self.use_coords:
            feats.append(pos)

        # 3. Geometric Triangulation Features (NEW)
        # We want to give the network explicit distances to the known anchors.
        # This helps it triangulate the unknown nodes.
        
        # Reshape to (B, N, 3)
        B = side.size(0)
        N = config.NUM_NODES
        pos_batch = pos.view(B, N, 3)
        
        # Extract Anchors: (B, 3, 3) -> 3 Anchors (0, 2, 7)
        anchors = pos_batch[:, config.KNOWN_IDS, :] 
        
        # Calculate Distance Matrix: (B, N, 3_anchors)
        # dist[b, n, a] = distance between node n and anchor a
        diff = pos_batch.unsqueeze(2) - anchors.unsqueeze(1) # (B, N, 3, 3)
        dists_to_anchors = torch.norm(diff, dim=-1) # (B, N, 3)
        
        # Calculate Distance to Centroid
        centroid = pos_batch.mean(dim=1, keepdim=True)
        dist_to_centroid = torch.norm(pos_batch - centroid, dim=-1, keepdim=True) # (B, N, 1)
        
        # Flatten and append
        geo_feats = torch.cat([dists_to_anchors, dist_to_centroid], dim=-1).view(N_total, -1)
        feats.append(geo_feats)

        # Concatenate all
        x = torch.cat(feats, dim=-1) 
        return self.mlp(x)