import torch
import pandas as pd
import numpy as np 
import config
from torch.geometric.data import Dataset, Data # type: ignore


def get_known_mask():
    """
    Create a mask for known landmarks coordinates
    Returns:
    - Mask for known landmark coordinates
    """
    mask = np.zeros(config.NUM_NODES, dtype=bool)
    mask[config.KNOWN_IDS] = True
    return mask

def extract_coords(row):
    """
    Fetch the 3D co-ordinate values for each landmark(node)

    Args:
        row : Series representing a single row at an index
    Returns:
        coordinates: (N,3) 3D co-ordinates of the landmarks(nodes)
    """
    coords = row.iloc[1, 1 + 3 * config.NUM_NODES].astype(float).values
    coords = coords.reshape(config.NUM_NODES, 3)
    return coords

def get_chirality(name: str):
    """
    Fetch chirality form subject name

    Args:
        name:(str) Subject name
    Returns:
        scalar chirality label
    """
    name = str(name).upper()
    return 0 if name.endswith("_L") else 1

class LoadFemurDataset(Dataset):
    def __init__(self, landmarks_csv_path: str, edges_csv_path: str, transform=None):
        """
        Args:
            landmarks_csv_path: (str) Path to landmarks CSV file
            edge_csv_path: (str) Path to edges CSV file
            transform: (bool) Transpose
        """
        super().__init__()
        self.landmarks_df = pd.read_csv(landmarks_csv_path, skiprows=[1])
        self.edges_df = pd.read_csv(edges_csv_path)
        self.transform = transform
        self.edge_index = torch.tensor(self.edges_df[["V1", "V2"]].values.T, dtype=torch.long)

    def __len__(self):
        return len(self.landmarks_df)

    def __getitem__(self, idx):
        """
        Returns:
            data: Dictionary containing:
                - coords: (N, 3) 3D landmarks(nodes) co-ordinates tensor
                - edge_index: (2,) Nodes connected by edges
                - edge_attr: (1, 52) Weights(eucledian distance) between edges
                - known_mask: (1,) Boolean mask for known co-ordinates
                - subject: (str) Subject name
        """
        row = self.landmarks_df.iloc[idx]
        subject = row['Source']
        coords = extract_coords(row=row)

        if subject not in self.edges_df.columns:
            raise KeyError(f"Subject: {subject} not found as a column in 'Edges.csv'")
        
        edge_attr = torch.tensor(self.edges_df.columns[subject].values.reshape(-1,1), dtype=torch.float32)
        side = get_chirality(subject)
        known_mask = torch.tensor(get_known_mask(), dtype=torch.bool)

        data = Data(
            coords = torch.tensor(coords, dtype=torch.float32),
            edge_index = self.edge_index,
            edge_attr = edge_attr,
            known_mask = known_mask, 
            subject = str(subject)
        )

        if self.transform:
            data = self.transform(data)
        
        return data
