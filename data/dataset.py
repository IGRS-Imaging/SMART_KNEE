# data/dataset.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
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
        self.augment = augment

        edge_idx = self.edges_df[["V1", "V2"]].values - 1
        self.edge_index = torch.tensor(edge_idx.T, dtype=torch.long)

    def __len__(self):
        return len(self.landmarks_df)

    def __getitem__(self, idx):
        row = self.landmarks_df.iloc[idx]
        subject = row["Source"]

        coords = extract_coords(row)
        pos = torch.tensor(coords, dtype=torch.float32)
        side = get_chirality(subject)

        # Augmentation: ONLY Left/Right flip (no random rotation now)
        if self.augment:
            if torch.rand(1) > 0.5:
                pos[:, 0] = -pos[:, 0]
                side = 1 - side

        if subject in self.edges_df.columns:
            dists = self.edges_df[subject].values.astype(float)
        else:
            dists = np.zeros(self.edge_index.size(1))

        edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(-1)

        data = Data(
            pos=pos,
            edge_index=self.edge_index.clone(),
            edge_attr=edge_attr,
            known_mask=torch.tensor(get_known_mask(), dtype=torch.bool),
            side=torch.tensor(side, dtype=torch.long),
            subject=str(subject),
            num_nodes=config.NUM_NODES,
        )
        return data


def get_dataloaders(landmarks_csv, edges_csv, batch_size, split=[0.8, 0.1, 0.1]):
    full_dataset = LoadFemurDataset(landmarks_csv, edges_csv, augment=False)
    total_size = len(full_dataset)

    train_size = int(total_size * split[0])
    val_size = int(total_size * split[1])
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_data = LoadFemurDataset(landmarks_csv, edges_csv, augment=False)
    val_data = LoadFemurDataset(landmarks_csv, edges_csv, augment=False)
    test_data = LoadFemurDataset(landmarks_csv, edges_csv, augment=False)

    train_subset.dataset = train_data
    val_subset.dataset = val_data
    test_subset.dataset = test_data

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
