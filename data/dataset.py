# data/dataset.py

import torch
import pandas as pd
import numpy as np
import config
from torch_geometric.data import Dataset, Data


def get_known_mask():
    mask = np.zeros(config.NUM_NODES, dtype=bool)
    mask[config.KNOWN_IDS] = True
    return mask


def extract_coords(row):
    """
    Row format after skipping 2nd header:
    Source, X0, Y0, Z0, X1, Y1, Z1, ..., X11, Y11, Z11, ...
    """
    vals = row.iloc[1 : 1 + 3 * config.NUM_NODES].astype(float).values
    coords = vals.reshape(config.NUM_NODES, 3)
    return coords


def get_chirality(name: str) -> int:
    name = str(name).upper()
    return 0 if name.endswith("_L") else 1


class LoadFemurDataset(Dataset):
    def __init__(self, landmarks_csv_path: str, edges_csv_path: str, transform=None):
        super().__init__()
        self.landmarks_df = pd.read_csv(landmarks_csv_path, skiprows=[1])
        self.edges_df = pd.read_csv(edges_csv_path)
        self.transform = transform

        # Build edge_index (0-based)
        edges = self.edges_df[["V1", "V2"]].values.astype(int)
        edges -= 1
        self.edge_index = torch.tensor(edges.T, dtype=torch.long)

    def __len__(self):
        return len(self.landmarks_df)

    def __getitem__(self, idx):
        row = self.landmarks_df.iloc[idx]
        subject = str(row["Source"])
        coords = extract_coords(row)

        if subject not in self.edges_df.columns:
            raise KeyError(f"Subject '{subject}' not found in Edges.csv")

        # ---- Edge attributes (per subject) ----
        edge_attr_vals = self.edges_df[subject].values.astype(float).reshape(-1, 1)

        # Normalize distances to [0, 1] range per subject
        max_val = np.max(edge_attr_vals)
        if max_val <= 0 or np.isnan(max_val):
            max_val = 1.0
        edge_attr_vals = edge_attr_vals / max_val

        edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32)

        data = Data(
            pos=torch.tensor(coords, dtype=torch.float32),           # (N,3)
            edge_index=self.edge_index,                              # (2,E)
            edge_attr=edge_attr,                                     # (E,1) normalized
            known_mask=torch.tensor(get_known_mask(), dtype=torch.bool),
            side=torch.tensor(get_chirality(subject), dtype=torch.long),
            subject=subject,
            num_nodes=config.NUM_NODES,
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
