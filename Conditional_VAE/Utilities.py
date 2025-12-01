import numpy as np
import pandas as pd
import torch
from Config import IN_DIM_COND

def extract_cond(row, in_dim=IN_DIM_COND):
    """Extract conditioning vector from CSV row."""
    slice_vals = row.iloc[1:1 + in_dim]
    cond_vals = pd.to_numeric(slice_vals, errors='coerce').fillna(0).values
    if len(cond_vals) < in_dim:
        cond_vals = np.pad(cond_vals, (0, in_dim - len(cond_vals)), 'constant')
    return torch.tensor(cond_vals, dtype=torch.float32)

def read_edges_csv(csv_path: str):
    """Read edges from CSV into edge_index tensor."""
    df = pd.read_csv(csv_path)
    if 'V1' not in df.columns or 'V2' not in df.columns:
        raise ValueError('Edges CSV must contain columns "V1" and "V2"')
    df['V1'] = pd.to_numeric(df['V1'], errors='coerce').fillna(0).astype(int)
    df['V2'] = pd.to_numeric(df['V2'], errors='coerce').fillna(0).astype(int)
    edge_arr = df[["V1", "V2"]].values - 1
    edge_index = torch.tensor(edge_arr.T, dtype=torch.long)
    return edge_index

def load_ply_points(path: str):
    """Load points from PLY file (unused if precomputed)."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)

def normalize_points(pts: np.ndarray):
    """Normalize points to unit sphere."""
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    scale = np.max(np.linalg.norm(pts_centered, axis=1)) or 1.0
    return pts_centered / scale, centroid, np.float32(scale)

def farthest_point_sampling_fast(points: np.ndarray, k: int):
    """Fast FPS for downsampling points."""
    N = points.shape[0]
    if N <= k:
        idx = np.arange(N, dtype=np.int64)
        if N < k:
            extras = np.random.choice(idx, size=k - N)
            idx = np.concatenate([idx, extras])
        return idx[:k]
    centroids = np.zeros(k, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float32)
    farthest = np.random.randint(0, N)
    for i in range(k):
        centroids[i] = farthest
        centroid_pt = points[farthest:farthest+1]
        dist = np.sum((points - centroid_pt) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return centroids

def chamfer_distance_raw(p1, p2):
    """Raw Chamfer distance (sum, not mean)."""
    dist_matrix = torch.cdist(p1, p2)
    cd1 = dist_matrix.min(dim=2)[0].sum(dim=1)
    cd2 = dist_matrix.min(dim=1)[0].sum(dim=1)
    return cd1 + cd2

def kl_raw(mu, logvar):
    """Raw KL divergence (per sample)."""
    var = torch.exp(logvar).clamp(min=1e-8)
    return -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=1)

def vae_loss_raw(recon, points, mu, logvar, kl_weight):
    """VAE loss with raw Chamfer + KL."""
    recon_raw = chamfer_distance_raw(recon, points)
    kl_raw_val = kl_raw(mu, logvar)
    total = recon_raw + kl_weight * kl_raw_val
    return total.mean(), recon_raw.mean().item(), kl_raw_val.mean().item()

