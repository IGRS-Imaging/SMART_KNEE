# models/encoder.py
import torch
import torch.nn as nn
import config


class Encoder(nn.Module):
    def __init__(self, feat_dim, hidden, use_coords=True):
        super().__init__()

        self.use_coords = use_coords

        # Embeddings
        self.chirality_emb = nn.Embedding(2, hidden)             # Left / Right
        self.node_index_emb = nn.Embedding(config.NUM_NODES, hidden)  # node id 0..NUM_NODES-1
        self.known_emb = nn.Embedding(2, hidden)                 # known = 1, unknown = 0

        coord_dim = 3 if use_coords else 0
        in_dim = hidden * 3 + coord_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, pos, known_mask, side, batch_idx):
        """
        pos:        (N_total, 3)
        known_mask: (N_total,)
        side:       (B,)
        batch_idx:  (N_total,) graph id for each node
        """
        device = pos.device
        N = pos.size(0)
        B = side.size(0)

        # Chirality embedding per node
        side_embed = self.chirality_emb(side)[batch_idx]     # (N, hidden)

        # Local node index embedding (0..NUM_NODES-1 for each graph)
        local_idx = torch.arange(N, device=device) % config.NUM_NODES
        node_embed = self.node_index_emb(local_idx)          # (N, hidden)

        # Known / unknown embedding
        known_embed = self.known_emb(known_mask.long())      # (N, hidden)

        feats = [side_embed, node_embed, known_embed]
        if self.use_coords:
            feats.append(pos)                                # (N, 3)

        x = torch.cat(feats, dim=-1)                         # (N, 3*hidden + coord_dim)
        return self.mlp(x)                                   # (N, feat_dim)
