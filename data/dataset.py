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

def kabsch_error(P, Q):
    Pc = np.mean(P, axis=0)
    Qc = np.mean(Q, axis=0)
    P_centered = P - Pc
    Q_centered = Q - Qc
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    return np.sqrt(np.mean(((P_centered @ R.T) - Q_centered)**2))

class LoadFemurDataset(Dataset):
    def __init__(self, landmarks_csv_path, edges_csv_path, augment=False):
        super().__init__()
        self.landmarks_df = pd.read_csv(landmarks_csv_path, skiprows=[1])
        self.edges_df = pd.read_csv(edges_csv_path)
        self.augment = augment

        # Check edge data
        landmark_subjects = set(self.landmarks_df["Source"].unique())
        edge_subjects = set(self.edges_df.columns)
        missing = len(landmark_subjects - edge_subjects)
        if missing > 0:
            print(f"⚠️ WARNING: {missing} subjects from Landmarks CSV are missing in Edges CSV!")

        edge_idx = self.edges_df[["V1", "V2"]].values - 1
        self.edge_index = torch.tensor(edge_idx.T, dtype=torch.long)

        try:
            self.mean_shape = np.load(config.MEAN_SHAPE_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing mean shape: {config.MEAN_SHAPE_PATH}")

    def __len__(self):
        return len(self.landmarks_df)

    def augment_samples(self, pos):
        """
        CRITICAL FIX: Reduced augmentation intensity
        - Less noise (0.3 -> 0.15mm)
        - Less scaling variation
        - More conservative rotations
        """
        # 1. Very small anatomical scaling (was 0.97-1.03)
        scale = np.random.uniform(0.98, 1.02)
        pos = pos * scale

        # 2. Small random rotation (was 0-2π, now much smaller)
        # Only rotate around primary axis to maintain anatomical orientation
        angle_range = np.pi / 12  # ±15 degrees
        angles = np.random.uniform(-angle_range, angle_range, size=3)
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        Rx = np.array([[1, 0, 0],
                    [0, cx, -sx],
                    [0, sx, cx]])
        Ry = np.array([[cy, 0, sy],
                    [0, 1, 0],
                    [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                    [sz, cz, 0],
                    [0, 0, 1]])

        R = Rz @ Ry @ Rx
        pos = pos @ R.T

        # 3. CRITICAL: Reduced noise (was 0.3, now 0.15mm)
        # This is the key fix - training data should match test data better
        noise = np.random.normal(0, 0.15, pos.shape)
        pos = pos + noise

        return pos

    def __getitem__(self, idx):
        row = self.landmarks_df.iloc[idx]
        subject = row["Source"]
        coords = extract_coords(row) 

        # Chirality Correction
        err_original = kabsch_error(coords, self.mean_shape)
        coords_flipped = coords.copy()
        coords_flipped[:, 0] *= -1
        err_flipped = kabsch_error(coords_flipped, self.mean_shape)
        
        is_flipped = False
        if err_flipped < err_original:
            coords = coords_flipped
            is_flipped = True
        
        side = 1 

        # CRITICAL: Apply augmentation AFTER chirality correction
        # This ensures augmented data maintains correct orientation
        if self.augment:
            coords = self.augment_samples(coords)
        
        pos = torch.tensor(coords, dtype=torch.float32)

        if subject in self.edges_df.columns:
            dists = self.edges_df[subject].values.astype(float)
        else:
            # Still use zeros if missing, but we've already warned
            dists = np.zeros(self.edge_index.size(1))

        edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(-1)

        data = Data(
            pos=pos,
            edge_index=self.edge_index.clone(),
            edge_attr=edge_attr,
            known_mask=torch.tensor(get_known_mask(), dtype=torch.bool),
            side=torch.tensor(side, dtype=torch.long),
            subject=str(subject),
            original_side=torch.tensor(0 if is_flipped else 1, dtype=torch.long), 
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
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_data = LoadFemurDataset(landmarks_csv, edges_csv, augment=True) 
    val_data = LoadFemurDataset(landmarks_csv, edges_csv, augment=False)
    test_data = LoadFemurDataset(landmarks_csv, edges_csv, augment=False)

    train_subset.dataset = train_data
    val_subset.dataset = val_data
    test_subset.dataset = test_data

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader