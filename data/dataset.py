import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
import config

def get_known_mask():
    mask = np.zeros(config.NUM_NODES, dtype=bool)
    mask[config.KNOWN_IDS] = True
    return mask

def extract_coords(row):
    coords = row.iloc[1: 1 + 3 * config.NUM_NODES].astype(float).values
    coords = coords.reshape(config.NUM_NODES, 3)
    return coords

def get_chirality(name):
    name = str(name).upper()
    return 0 if name.endswith("_L") else 1

class LoadFemurDataset(Dataset):
    def __init__(self, landmarks_csv_path, edges_csv_path, augment=False):
        super().__init__()
        self.landmarks_df = pd.read_csv(landmarks_csv_path, skiprows=[1])
        self.edges_df = pd.read_csv(edges_csv_path)
        
        # Create edge index (0-based)
        edge_idx = self.edges_df[["V1", "V2"]].values - 1
        self.edge_index = torch.tensor(edge_idx.T, dtype=torch.long)

    def __len__(self):
        return len(self.landmarks_df)

    def __getitem__(self, idx):
        row = self.landmarks_df.iloc[idx]
        subject = row["Source"]

        # 1. Get Positions
        coords = extract_coords(row)
        
        # 2. Get Edge Attributes (Target Distances)
        # Assumes edges_df has columns named by Subject ID containing distances
        # Shape: (Num_Edges, 1)
        if subject in self.edges_df.columns:
            dists = self.edges_df[subject].values.astype(float)
        else:
            # Fallback if subject not found (should ensure data integrity)
            dists = np.zeros(self.edge_index.size(1))
            
        edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(-1)

        known_mask = torch.tensor(get_known_mask(), dtype=torch.bool)
        side = torch.tensor(get_chirality(subject), dtype=torch.long)


        data = Data(
            pos=torch.tensor(coords, dtype=torch.float32),
            edge_index=self.edge_index.clone(),
            edge_attr=edge_attr, 
            known_mask=known_mask,
            side=side,
            subject=str(subject),
            num_nodes=config.NUM_NODES
        )
        return data
