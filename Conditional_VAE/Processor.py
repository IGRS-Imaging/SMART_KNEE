import torch.nn as nn
from Config import COND_EMB_DIM, IN_DIM_COND

class Processor(nn.Module):
    """Processes conditioning input to embedding."""
    def __init__(self, in_dim=IN_DIM_COND, emb_dim=COND_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim), nn.ReLU()
        )

    def forward(self, cond):
        return self.net(cond)                                                                                                                                                                                                                                                                                                 