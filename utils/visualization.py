# utils/visualization.py

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_shapes(pos_gt, pos_init, pos_pred, edge_index, title="Shape Visualization"):
    """
    pos_gt:   (N,3) ground truth coords
    pos_init: (N,3) initialized coords (centroid init)
    pos_pred: (N,3) predicted coords from EGNN
    edge_index: (2,E) graph structure
    """

    fig = plt.figure(figsize=(12, 6))

    # ---------------------------
    # 1) 3D Ground Truth
    # ---------------------------
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title("Ground Truth")
    ax1.scatter(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], c="green", s=40, label="GT")

    # draw edges
    for u, v in edge_index.t().tolist():
        ax1.plot(
            [pos_gt[u, 0], pos_gt[v, 0]],
            [pos_gt[u, 1], pos_gt[v, 1]],
            [pos_gt[u, 2], pos_gt[v, 2]],
            c="gray",
            linewidth=1
        )

    ax1.legend()

    # ---------------------------
    # 2) Initialization (centroid-based)
    # ---------------------------
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_title("Initialization (centroid init)")

    ax2.scatter(pos_init[:, 0], pos_init[:, 1], pos_init[:, 2],
                c="orange", s=40, label="Init")

    for u, v in edge_index.t().tolist():
        ax2.plot(
            [pos_init[u, 0], pos_init[v, 0]],
            [pos_init[u, 1], pos_init[v, 1]],
            [pos_init[u, 2], pos_init[v, 2]],
            c="gray",
            linewidth=1
        )

    ax2.legend()

    # ---------------------------
    # 3) Predicted Shape
    # ---------------------------
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_title("Prediction")

    ax3.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                c="blue", s=40, label="Pred")

    for u, v in edge_index.t().tolist():
        ax3.plot(
            [pos_pred[u, 0], pos_pred[v, 0]],
            [pos_pred[u, 1], pos_pred[v, 1]],
            [pos_pred[u, 2], pos_pred[v, 2]],
            c="gray",
            linewidth=1
        )

    ax3.legend()

    # ---------------------------
    # Global layout
    # ---------------------------
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
