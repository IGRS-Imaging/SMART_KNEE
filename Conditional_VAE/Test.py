# import os
# import torch
# import numpy as np
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# try:
#     import open3d as o3d
#     O3D_AVAILABLE = True
# except ImportError:
#     O3D_AVAILABLE = False
#     print("open3d not installed; skipping .ply saves")

# try:
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#     PLOTLY_AVAILABLE = True
# except ImportError:
#     PLOTLY_AVAILABLE = False
#     print("Plotly not installed; using matplotlib")

# from Config import ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, DEVICE, CHECKPOINT_DIR
# from Dataloader import FemurVAEDataset, collate_fn
# from Model import FemurVAE
# from torch.utils.data import DataLoader

# def points_to_ply(points, filename):
#     if not O3D_AVAILABLE:
#         print(f"Skipping {filename} (open3d required)")
#         return
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.io.write_point_cloud(filename, pcd)
#     print(f"Saved {filename}")

# def visualize_3d_plotly(points_list, titles, filename='viz.html'):
#     if not PLOTLY_AVAILABLE:
#         print("Plotly unavailable; using matplotlib fallback")
#         fallback_matplotlib_viz(points_list, titles)
#         return
    
#     fig = make_subplots(rows=1, cols=len(points_list), 
#                         subplot_titles=titles, specs=[[{"type": "scatter3d"}]*len(points_list)])
    
#     colors = ['blue', 'red', 'green', 'orange', 'purple']
#     for i, (pts, title, color) in enumerate(zip(points_list, titles, colors[:len(points_list)])):
#         fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], 
#                                    mode='markers', marker=dict(size=3, color=color),
#                                    name=title), row=1, col=i+1)
    
#     fig.update_layout(height=600, title_text="3D Point Clouds (mm scale)", scene=dict(aspectmode='data'))
#     fig.write_html(filename)
#     fig.show()
#     print(f"Saved interactive viz to {filename}")

# def fallback_matplotlib_viz(points_list, titles):
#     fig = plt.figure(figsize=(5 * len(points_list), 5))
#     for i, (pts, title) in enumerate(zip(points_list, titles)):
#         ax = fig.add_subplot(1, len(points_list), i+1, projection='3d')
#         ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=20)
#         ax.set_title(title)
#         ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
#     plt.tight_layout()
#     plt.savefig('static_viz.png', dpi=200)
#     plt.show()
#     print("Saved static viz to static_viz.png")

# def denormalize(recon_norm, pts_mean, pts_std):
#     """Denorm recon using input stats."""
#     return recon_norm * pts_std + pts_mean

# def test_reconstructions(model_path='vae_best.pt', num_samples=3):
#     ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV)
#     loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

#     model = FemurVAE(n_points=1024, latent_dim=256, cond_dim=6).to(DEVICE)  # Match train
#     model.load_state_dict(
#         torch.load(os.path.join(CHECKPOINT_DIR, model_path), map_location=DEVICE, weights_only=True)
#     )
#     model.eval()

#     points_list = []
#     titles = []
#     total_rmse = 0.0
#     num_processed = 0

#     print("Running reconstruction test...")
#     with torch.no_grad():
#         pbar = tqdm(enumerate(loader), total=num_samples, desc="Recon Samples")
#         for i, batch in pbar:
#             if i >= num_samples:
#                 break
#             pts = batch['points'].to(DEVICE)
#             cond = batch['cond'].to(DEVICE)

#             # Norm for model
#             pts_mean = pts.mean(dim=1, keepdim=True)
#             pts_std = pts.std(dim=1, keepdim=True) + 1e-6
#             pts_norm = (pts - pts_mean) / pts_std
#             cond_mean, cond_std = 0.0, 1.0  # From train
#             cond_norm = (cond - cond_mean) / (cond_std + 1e-6)

#             # Reconstruct
#             recon_norm, _, _ = model(pts_norm, cond_norm)
#             recon = denormalize(recon_norm, pts_mean, pts_std)  # Denorm for viz/RMSE
#             orig = pts.cpu().numpy().squeeze()
#             recon_np = recon.cpu().numpy().squeeze()

#             # Debug
#             print(f"Sample {i+1}: pts shape={pts.shape}, recon shape={recon.shape}")

#             # RMSE on denorm (mm)
#             rmse = np.sqrt(np.mean((orig - recon_np)**2))
#             total_rmse += rmse
#             num_processed += 1
#             pbar.set_postfix({'RMSE (mm)': f"{rmse:.2f}"})

#             points_list.extend([orig, recon_np])
#             titles.extend([f"Orig {i+1}", f"Recon {i+1}"])

#             points_to_ply(recon_np, f"recon_{i+1}.ply")
#             print(f"Sample {i+1} RMSE: {rmse:.2f} mm")

#     avg_rmse = total_rmse / num_processed if num_processed > 0 else 0
#     print(f"Average RMSE over {num_processed} samples: {avg_rmse:.2f} mm")

#     visualize_3d_plotly(points_list, titles)

# def test_generation(model_path='vae_best.pt', num_samples=5, custom_cond=None, save_ply=True):
#     model = FemurVAE(n_points=1024, latent_dim=256, cond_dim=6).to(DEVICE)
#     model.load_state_dict(
#         torch.load(os.path.join(CHECKPOINT_DIR, model_path), map_location=DEVICE, weights_only=True)
#     )
#     model.eval()

#     latent_dim = model.latent_dim
#     cond_dim = model.cond_dim
#     n_points = model.n_points

#     points_list = []
#     titles = []

#     print("Running generation test...")
#     with torch.no_grad():
#         pbar = tqdm(range(num_samples), desc="Gen Samples")
#         for i in pbar:
#             z = torch.randn(1, latent_dim).to(DEVICE)

#             if custom_cond is not None and len(custom_cond) == cond_dim:
#                 cond = torch.tensor([custom_cond], dtype=torch.float32).to(DEVICE)
#             else:
#                 cond = (torch.rand(1, cond_dim).to(DEVICE) - 0.5) * 2

#             # Norm cond
#             cond_mean, cond_std = 0.0, 1.0
#             cond_norm = (cond - cond_mean) / (cond_std + 1e-6)

#             # Generate with decode (no dummy!)
#             gen_norm = model.decode(z, cond_norm)
            
#             # Dummy denorm for viz (use avg femur stats; improve with real mean/std from ds)
#             dummy_mean = torch.tensor([[0.0, 0.0, 0.0]]).to(DEVICE)  # Avg center
#             dummy_std = torch.tensor([[100.0, 100.0, 100.0]]).to(DEVICE)  # Avg scale ~100mm
#             gen = denormalize(gen_norm, dummy_mean, dummy_std)
#             gen_np = gen.cpu().numpy().squeeze()

#             print(f"Gen {i+1}: gen shape={gen.shape}")

#             points_list.append(gen_np)
#             title = f"Generated {i+1}"
#             if custom_cond:
#                 title += f" (cond: {custom_cond[:3]}...)"
#             else:
#                 title += f" (rand cond: {cond.cpu().numpy().flatten()[:3]}...)"
#             titles.append(title)

#             if save_ply:
#                 points_to_ply(gen_np, f"generated_{i+1}.ply")

#             print(f"Generated sample {i+1} with cond {cond.cpu().numpy().flatten()}")
#             pbar.set_postfix({'cond_preview': f"{cond.cpu().numpy().flatten()[:2]}..."})

#     visualize_3d_plotly(points_list, titles)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', default='recon', choices=['recon', 'generate'], help='Test mode')
#     parser.add_argument('--model_path', default='vae_best.pt', help='Checkpoint path')
#     parser.add_argument('--num_samples', type=int, default=5, help='Num examples')
#     parser.add_argument('--custom_cond', nargs='+', type=float, default=None, 
#                         help='Custom scalars e.g., --custom_cond 0.5 120.0 45.0 10.0 15.0 20.0')
#     parser.add_argument('--save_ply', action='store_true', help='Save .ply files')
#     args = parser.parse_args()

#     if args.mode == 'recon':
#         test_reconstructions(args.model_path, args.num_samples)
#     else:
#         test_generation(args.model_path, args.num_samples, args.custom_cond if args.custom_cond else None, args.save_ply)


import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("open3d not installed; skipping .ply saves")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed; using matplotlib")

from Config import ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, DEVICE, CHECKPOINT_DIR
from Dataloader import FemurVAEDataset, collate_fn
from Model import FemurVAE
from torch.utils.data import DataLoader

def points_to_ply(points, filename):
    if not O3D_AVAILABLE:
        print(f"Skipping {filename} (open3d required)")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {filename}")

def visualize_3d_plotly(points_list, titles, filename='viz.html'):
    if not PLOTLY_AVAILABLE:
        print("Plotly unavailable; using matplotlib fallback")
        fallback_matplotlib_viz(points_list, titles)
        return
    
    fig = make_subplots(rows=1, cols=len(points_list), 
                        subplot_titles=titles, specs=[[{"type": "scatter3d"}]*len(points_list)])
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (pts, title, color) in enumerate(zip(points_list, titles, colors[:len(points_list)])):
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], 
                                   mode='markers', marker=dict(size=3, color=color),
                                   name=title), row=1, col=i+1)
    
    fig.update_layout(height=600, title_text="3D Point Clouds (mm scale)", scene=dict(aspectmode='data'))
    fig.write_html(filename)
    fig.show()
    print(f"Saved interactive viz to {filename}")

def fallback_matplotlib_viz(points_list, titles):
    fig = plt.figure(figsize=(5 * len(points_list), 5))
    for i, (pts, title) in enumerate(zip(points_list, titles)):
        ax = fig.add_subplot(1, len(points_list), i+1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=20)
        ax.set_title(title)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    plt.tight_layout()
    plt.savefig('static_viz.png', dpi=200)
    plt.show()
    print("Saved static viz to static_viz.png")

def denormalize(recon_norm, pts_mean, pts_std, cond_mean, cond_std):
    recon = recon_norm * pts_std + pts_mean  # Points denorm
    return recon

def test_reconstructions(model_path='vae_best.pt', num_samples=3):
    ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = FemurVAE(n_points=1024, latent_dim=256, cond_dim=42).to(DEVICE)  # FIXED: 42
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, model_path), map_location=DEVICE, weights_only=True)
    )
    model.eval()

    # Cond stats (from train or compute)
    all_cond = torch.stack([torch.tensor(ds[i]['cond']) for i in range(len(ds))])
    cond_mean = all_cond.mean(dim=0)
    cond_std = all_cond.std(dim=0) + 1e-6

    points_list = []
    titles = []
    total_rmse = 0.0
    num_processed = 0

    print("Running reconstruction test...")
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=num_samples, desc="Recon Samples")
        for i, batch in pbar:
            if i >= num_samples:
                break
            pts = batch['points'].to(DEVICE)
            cond = batch['cond'].to(DEVICE)

            # Norm
            pts_mean = pts.mean(dim=1, keepdim=True)
            pts_std = pts.std(dim=1, keepdim=True) + 1e-6
            pts_norm = (pts - pts_mean) / pts_std
            cond_norm = (cond - cond_mean.to(DEVICE)) / cond_std.to(DEVICE)

            # Reconstruct
            recon_norm, _, _ = model(pts_norm, cond_norm)
            recon = denormalize(recon_norm, pts_mean, pts_std, cond_mean.to(DEVICE), cond_std.to(DEVICE))
            orig = pts.cpu().numpy().squeeze()
            recon_np = recon.cpu().numpy().squeeze()

            print(f"Sample {i+1}: pts shape={pts.shape}, recon shape={recon.shape}")

            # RMSE (mm)
            rmse = np.sqrt(np.mean((orig - recon_np)**2))
            total_rmse += rmse
            num_processed += 1
            pbar.set_postfix({'RMSE (mm)': f"{rmse:.2f}"})

            points_list.extend([orig, recon_np])
            titles.extend([f"Orig {i+1}", f"Recon {i+1}"])

            points_to_ply(recon_np, f"recon_{i+1}.ply")
            print(f"Sample {i+1} RMSE: {rmse:.2f} mm")

    avg_rmse = total_rmse / num_processed if num_processed > 0 else 0
    print(f"Average RMSE over {num_processed} samples: {avg_rmse:.2f} mm")

    visualize_3d_plotly(points_list, titles)

def test_generation(model_path='vae_best.pt', num_samples=5, custom_cond=None, save_ply=True):
    model = FemurVAE(n_points=1024, latent_dim=256, cond_dim=42).to(DEVICE)  # FIXED: 42
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, model_path), map_location=DEVICE, weights_only=True)
    )
    model.eval()

    latent_dim = model.latent_dim
    cond_dim = model.cond_dim
    n_points = model.n_points

    # Cond stats
    ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV)
    all_cond = torch.stack([torch.tensor(ds[i]['cond']) for i in range(min(100, len(ds)))])  # Sample for speed
    cond_mean = all_cond.mean(dim=0)
    cond_std = all_cond.std(dim=0) + 1e-6

    points_list = []
    titles = []

    print("Running generation test...")
    with torch.no_grad():
        pbar = tqdm(range(num_samples), desc="Gen Samples")
        for i in pbar:
            z = torch.randn(1, latent_dim).to(DEVICE)

            if custom_cond is not None and len(custom_cond) == cond_dim:
                cond = torch.tensor([custom_cond], dtype=torch.float32).to(DEVICE)
            else:
                cond = (torch.rand(1, cond_dim).to(DEVICE) - 0.5) * 2

            cond_norm = (cond - cond_mean.to(DEVICE)) / cond_std.to(DEVICE)

            # Gen with decode
            gen_norm = model.decode(z, cond_norm)
            
            # Avg denorm
            avg_pts_mean = torch.zeros(1, 1, 3).to(DEVICE)
            avg_pts_std = torch.ones(1, 1, 3).to(DEVICE) * 100.0  # ~femur scale
            gen = denormalize(gen_norm, avg_pts_mean, avg_pts_std, cond_mean.to(DEVICE), cond_std.to(DEVICE))
            gen_np = gen.cpu().numpy().squeeze()

            print(f"Gen {i+1}: gen shape={gen.shape}")

            points_list.append(gen_np)
            title = f"Generated {i+1}"
            if custom_cond:
                title += f" (cond: {custom_cond[:3]}...)"
            else:
                title += f" (rand cond: {cond.cpu().numpy().flatten()[:3]}...)"
            titles.append(title)

            if save_ply:
                points_to_ply(gen_np, f"generated_{i+1}.ply")

            print(f"Generated sample {i+1} with cond {cond.cpu().numpy().flatten()}")
            pbar.set_postfix({'cond_preview': f"{cond.cpu().numpy().flatten()[:2]}..."})

    visualize_3d_plotly(points_list, titles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='recon', choices=['recon', 'generate'], help='Test mode')
    parser.add_argument('--model_path', default='vae_best.pt', help='Checkpoint path')
    parser.add_argument('--num_samples', type=int, default=5, help='Num examples')
    parser.add_argument('--custom_cond', nargs='+', type=float, default=None, 
                        help='Custom scalars e.g., --custom_cond 0.5 120.0 45.0 ... (42 values)')
    parser.add_argument('--save_ply', action='store_true', help='Save .ply files')
    args = parser.parse_args()

    if args.mode == 'recon':
        test_reconstructions(args.model_path, args.num_samples)
    else:
        test_generation(args.model_path, args.num_samples, args.custom_cond if args.custom_cond else None, args.save_ply)