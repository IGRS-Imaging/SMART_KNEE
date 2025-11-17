# data/dataset.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
import config


# ---------------------------------------------
# 1. Known landmark mask (global for all femurs)
# ---------------------------------------------
def get_known_mask():
    mask = np.zeros(config.NUM_NODES, dtype=bool)
    mask[config.KNOWN_IDS] = True
    return mask


# ---------------------------------------------
# 2. Extract coords correctly
# ---------------------------------------------
def extract_coords(row):
    coords = row.iloc[1: 1 + 3 * config.NUM_NODES].astype(float).values
    coords = coords.reshape(config.NUM_NODES, 3)
    return coords


# ---------------------------------------------
# 3. Chirality from name
# ---------------------------------------------
def get_chirality(name):
    name = str(name).upper()
    return 0 if name.endswith("_L") else 1


# ---------------------------------------------
# 4. Random rotation augmentation
# ---------------------------------------------
def random_rotation_matrix():
    # sample random rotation
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    psi = np.random.uniform(0, 2*np.pi)

    Rz1 = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

    Ry  = np.array([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]])

    Rz2 = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])

    return Rz2 @ Ry @ Rz1


# ---------------------------------------------
# 5. Dataset class
# ---------------------------------------------
class LoadFemurDataset(Dataset):

    def __init__(self, landmarks_csv_path, edges_csv_path, augment=True):
        super().__init__()

        self.landmarks_df = pd.read_csv(landmarks_csv_path, skiprows=[1])
        self.edges_df = pd.read_csv(edges_csv_path)

        self.augment = augment

        # edge index: convert 1-based → 0-based
        edge_idx = self.edges_df[["V1", "V2"]].values - 1
        self.edge_index = torch.tensor(edge_idx.T, dtype=torch.long)

        self.edge_attr = None  # REMOVE edge features (EGNN computes distance internally)

    def __len__(self):
        return len(self.landmarks_df)

    def __getitem__(self, idx):
        row = self.landmarks_df.iloc[idx]
        subject = row["Source"]

        coords = extract_coords(row)
        known_mask = torch.tensor(get_known_mask(), dtype=torch.bool)
        side = torch.tensor(get_chirality(subject), dtype=torch.long)

        # -------------------------------
        # Apply random rotation augmentation
        # -------------------------------
        if self.augment:
            R = random_rotation_matrix()
            coords = coords @ R.T

        data = Data(
            pos=torch.tensor(coords, dtype=torch.float32),
            edge_index=self.edge_index.clone(),
            known_mask=known_mask,
            side=side,
            subject=str(subject),
            num_nodes=config.NUM_NODES
        )

        return data
