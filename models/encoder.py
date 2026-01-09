# models/encoder.py
import torch
import torch.nn as nn
import config

class Encoder(nn.Module):
    def __init__(self, feat_dim, hidden, use_coords=True):
        super().__init__()
        self.use_coords = use_coords

        # Embeddings
        self.chirality_emb = nn.Embedding(2, hidden)
        self.node_index_emb = nn.Embedding(config.NUM_NODES, hidden)
        self.known_emb = nn.Embedding(2, hidden)

        coord_dim = 3 if use_coords else 0
        num_anchors = len(config.KNOWN_IDS)
        # ENHANCED GEOMETRIC FEATURES
        # 1. Distances to anchors (3)
        # 2. Distance to centroid (1)
        # 3. Relative position to each anchor (3*3=9) - NEW
        # 4. Distance variance (1) - NEW
        # 5. Bone axis projection (2) - NEW
        geo_dim = num_anchors + 1 + (num_anchors * 3) + 1 + 2
        
        in_dim = hidden * 3 + coord_dim + geo_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.15),  # Reduced from 0.2
            nn.Linear(hidden, hidden * 2),  # Wider layer
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, pos, known_mask, side, batch_idx):
        """
        pos: (N_total, 3) - Initial positions
        """
        device = pos.device
        N_total = pos.size(0)
        
        # 1. Basic Embeddings
        side_embed = self.chirality_emb(side)[batch_idx]
        local_idx = torch.arange(N_total, device=device) % config.NUM_NODES
        node_embed = self.node_index_emb(local_idx)
        known_embed = self.known_emb(known_mask.long())

        feats = [side_embed, node_embed, known_embed]

        # 2. Coordinate Features
        if self.use_coords:
            feats.append(pos)

        # 3. ENHANCED GEOMETRIC FEATURES
        B = side.size(0)
        N = config.NUM_NODES
        pos_batch = pos.view(B, N, 3)
        
        # Extract Anchors: (B, 3, 3)
        anchors = pos_batch[:, config.KNOWN_IDS, :]
        
        # A. Distance to each anchor (3 features)
        diff_to_anchors = pos_batch.unsqueeze(2) - anchors.unsqueeze(1)  # (B, N, 3, 3)
        dists_to_anchors = torch.norm(diff_to_anchors, dim=-1)  # (B, N, 3)
        
        # B. Distance to centroid (1 feature)
        centroid = pos_batch.mean(dim=1, keepdim=True)
        dist_to_centroid = torch.norm(pos_batch - centroid, dim=-1, keepdim=True)
        
        # C. Relative position to each anchor (9 features) - NEW
        # This gives directional information, not just distance
        rel_pos_to_anchors = diff_to_anchors.view(B, N, -1)  # (B, N, 9)
        
        # D. Distance variance (1 feature) - NEW
        # How "spread out" is this node from the anchors?
        dist_variance = dists_to_anchors.var(dim=-1, keepdim=True)
        
        # E. Bone axis projection (2 features) - NEW
        # Project onto principal bone axis (defined by anchor 0 to anchor 2)
        bone_axis = anchors[:, 2, :] - anchors[:, 0, :]  # (B, 3)
        bone_axis = bone_axis / (torch.norm(bone_axis, dim=-1, keepdim=True) + 1e-8)
        bone_axis = bone_axis.unsqueeze(1)  # (B, 1, 3)
        
        # Project each point onto this axis
        pos_centered = pos_batch - anchors[:, 0:1, :]  # Center at anchor 0
        proj_on_axis = (pos_centered * bone_axis).sum(dim=-1, keepdim=True)  # (B, N, 1)
        
        # Distance from axis (perpendicular component)
        parallel_component = proj_on_axis * bone_axis
        perp_component = pos_centered - parallel_component
        dist_from_axis = torch.norm(perp_component, dim=-1, keepdim=True)  # (B, N, 1)
        
        bone_axis_features = torch.cat([proj_on_axis, dist_from_axis], dim=-1)  # (B, N, 2)
        
        # Combine all geometric features
        geo_feats = torch.cat([
            dists_to_anchors,          # (B, N, 3)
            dist_to_centroid,          # (B, N, 1)
            rel_pos_to_anchors,        # (B, N, 9)
            dist_variance,             # (B, N, 1)
            bone_axis_features         # (B, N, 2)
        ], dim=-1).view(N_total, -1)  # (N_total, 16)
        
        feats.append(geo_feats)

        # Concatenate all
        x = torch.cat(feats, dim=-1)
        return self.mlp(x)