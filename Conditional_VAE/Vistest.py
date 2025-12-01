import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Config import ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, DEVICE, CHECKPOINT_DIR
from Dataloader import FemurVAEDataset, collate_fn
from Model import FemurVAE
from torch.utils.data import DataLoader

def visualize_reconstructions(model_path=r"C:\Users\sweth\Downloads\Conditional_VAE\checkpoints\vae_best.pt", num_samples=5):
    # Load dataset (use full or val split)
    ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)  # Batch=1 for viz

    # Load model
    model = FemurVAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    fig = plt.figure(figsize=(15, 3 * num_samples))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            pts = batch['points'].to(DEVICE)  # Shape: (1, N_points, 3)
            cond = batch['cond'].to(DEVICE)   # Condition (e.g., pose/scale)

            # Reconstruct
            recon, _, _ = model(pts, cond)
            orig = pts.cpu().numpy().squeeze()
            recon_np = recon.cpu().numpy().squeeze()

            # Plot 3D scatter
            ax = fig.add_subplot(num_samples, 3, 3*i + 1, projection='3d')
            ax.scatter(orig[:, 0], orig[:, 1], orig[:, 2], c='blue', s=50, label='Original')
            ax.set_title(f'Sample {i+1} - Original')
            ax.legend()

            ax = fig.add_subplot(num_samples, 3, 3*i + 2, projection='3d')
            ax.scatter(recon_np[:, 0], recon_np[:, 1], recon_np[:, 2], c='red', s=50, label='Reconstructed')
            ax.set_title(f'Sample {i+1} - Reconstructed')
            ax.legend()

            # Error vectors (optional: plot differences)
            diff = orig - recon_np
            ax = fig.add_subplot(num_samples, 3, 3*i + 3, projection='3d')
            ax.quiver(orig[:, 0], orig[:, 1], orig[:, 2],
                      diff[:, 0], diff[:, 1], diff[:, 2], color='green', arrow_length_ratio=0.1)
            ax.scatter(orig[:, 0], orig[:, 1], orig[:, 2], c='blue', s=20)
            ax.set_title(f'Sample {i+1} - Error Vectors')
            rmse = np.sqrt(np.mean(diff**2))
            ax.text2D(0.05, 0.95, f'RMSE: {rmse:.2f}mm', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('recon_viz.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved recon_viz.png")

if __name__ == '__main__':
    visualize_reconstructions()