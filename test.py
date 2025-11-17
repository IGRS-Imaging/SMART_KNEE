# test.py
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

import config
from data.dataset import LoadFemurDataset
from models import LandmarkCompletionModel
from losses import procrustes_align
from utils.visualization import visualize_shapes


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on device:", device)

    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = LandmarkCompletionModel().to(device)
    model.load_state_dict(torch.load(config.TEST_MODEL_PATH, map_location=device))
    model.eval()

    errors = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = batch.to(device)

            pred, pos_init = model(batch)

            pred = pred.view(1, config.NUM_NODES, 3)
            target = batch.pos.view(1, config.NUM_NODES, 3)

            # reshape known mask
            known = batch.known_mask.view(1, config.NUM_NODES)

            # Procrustes alignment
            pred_a, _, _ = procrustes_align(pred, target)

            # Only unknown nodes contribute to error
            unknown = (~known).float().unsqueeze(-1)
            err = ((pred_a - target) * unknown).norm(dim=-1).mean()
            errors.append(err.item())

            # visualize the first sample
            if idx == 0:
                visualize_shapes(
                    pos_gt=target[0].cpu(),
                    pos_init=pos_init.view(config.NUM_NODES, 3).cpu(),
                    pos_pred=pred.view(config.NUM_NODES, 3).cpu(),
                    edge_index=batch.edge_index.cpu(),
                    title=f"Prediction (Sample {idx})"
                )

    # ---------- PLOT P-MPJPE ----------
    plt.figure(figsize=(8, 5))
    plt.plot(errors, marker='o', linewidth=2)
    plt.title("P-MPJPE per Test Sample", fontsize=16)
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("P-MPJPE (mm)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    # -----------------------------------

    print(f"\nMean P-MPJPE: {sum(errors)/len(errors):.3f} mm")


if __name__ == "__main__":
    test()
