import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from Config import NUM_POINTS, IN_DIM_COND
from Utilities import extract_cond, read_edges_csv

class FemurVAEDataset(Dataset):
    """Dataset for femur point clouds with conditioning."""
    def __init__(self, ROOT_dir: str, landmarks_csv: str, edges_csv: str, num_points: int = NUM_POINTS, indices=None):
        self.ROOT = Path(ROOT_dir)
        all_subfolders = sorted([p for p in self.ROOT.iterdir() if p.is_dir()])
        self.subfolders = [all_subfolders[i] for i in indices] if indices is not None else all_subfolders
        self.num_points = num_points

        # Read landmarks and build subject-to-cond mapping
        self.landmarks_df = pd.read_csv(landmarks_csv)
        self.subject_to_cond = {}           
        for _, row in self.landmarks_df.iterrows():
            subject = str(row.iloc[0]).strip()
            self.subject_to_cond[subject] = extract_cond(row)

        # Read edges
        self.edge_index = read_edges_csv(edges_csv)

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        folder = self.subfolders[idx]
        subj = folder.name
        cond = self.subject_to_cond.get(subj, torch.zeros(IN_DIM_COND, dtype=torch.float32))

        # Load precomputed .npy files
        pts_sampled = np.load(folder / "precomputed_points.npy")
        centroid = np.load(folder / "precomputed_centroid.npy")
        scale_raw = np.load(folder / "precomputed_scale.npy", allow_pickle=True)
        try:
            scale = float(scale_raw) if np.isscalar(scale_raw) else float(np.array(scale_raw).item())
        except Exception:
            scale = float(np.atleast_1d(scale_raw)[0])

        pts_tensor = torch.from_numpy(pts_sampled.astype(np.float32))

        return {
            "points": pts_tensor,
            "cond": cond,
            "subject_id": subj,
            "centroid": torch.from_numpy(centroid.astype(np.float32)),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "edge_index": self.edge_index
        }

# def collate_fn(batch):
#     """Custom collate for batching."""
#     pts = torch.stack([b['points'] for b in batch])
#     cond = torch.stack([b['cond'] for b in batch])
#     subjects = [b['subject_id'] for b in batch]
#     centroids = torch.stack([b['centroid'] for b in batch])
#     scales = torch.stack([b['scale'] for b in batch])
#     edge_index = batch[0]['edge_index'] if 'edge_index' in batch[0] else None
#     return {"points": pts, "cond": cond, "subject_id": subjects, "centroid": centroids, "scale": scales, "edge_index": edge_index}

def collate_fn(batch):
    points = torch.stack([item['points'] for item in batch])  # (B, 1024, 3)
    conds = torch.stack([item['cond'] for item in batch])    # (B, 6)
    return {'points': points, 'cond': conds}