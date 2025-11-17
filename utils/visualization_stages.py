import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_stage(ax, pts, title, color):
    if torch.is_tensor(pts):
        pts = pts.cpu().numpy()
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def visualize_stages(raw, init, norm, pred):
    fig = plt.figure(figsize=(14,14))

    ax1 = fig.add_subplot(221, projection="3d")
    plot_stage(ax1, raw, "1. Raw Input", "blue")

    ax2 = fig.add_subplot(222, projection="3d")
    plot_stage(ax2, init, "2. After Unknown Initialization", "orange")

    ax3 = fig.add_subplot(223, projection="3d")
    plot_stage(ax3, norm, "3. Normalized Coords", "green")

    ax4 = fig.add_subplot(224, projection="3d")
    plot_stage(ax4, pred, "4. Final Prediction", "red")

    plt.show()
