# test.py

import torch
from torch_geometric.loader import DataLoader

import config
from data import LoadFemurDataset
from models import LandmarkCompletionModel
from losses import procrustes_align
from utils.visualization import visualize_shapes


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on device:", device)

    # ----------------------------
    # Load dataset
    # ----------------------------
    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # ----------------------------
    # Load model
    # ----------------------------
    model = LandmarkCompletionModel().to(device)
    model.load_state_dict(torch.load(config.TEST_MODEL_PATH, map_location=device))
    model.eval()

    all_errors = []

    # ----------------------------
    # Inference loop
    # ----------------------------
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            batch = batch.to(device)

            # Model outputs: pred_flat, pos_init_flat
            pred_flat, pos_init_flat = model(batch)

            B = batch.num_graphs
            N = config.NUM_NODES

            pred = pred_flat.view(B, N, 3)
            pos_init = pos_init_flat.view(B, N, 3)
            target = batch.pos.view(B, N, 3)

            # ----------------------------
            # Error after Procrustes alignment
            # ----------------------------
            pred_aligned, _, _ = procrustes_align(pred, target)
            error = (pred_aligned - target).norm(dim=-1).mean(dim=-1)  # (B,)
            all_errors.append(error.cpu())

            # ----------------------------
            # Visualize only the FIRST sample of FIRST batch
            # ----------------------------
            if batch_idx == 0:
                graph_mask = batch.batch == 0       # boolean mask for graph 0
                node_ids = graph_mask.nonzero(as_tuple=True)[0]
                
                # Extract nodes belonging to first graph
                pos_gt_single = target[0]
                pos_init_single = pos_init[0]
                pos_pred_single = pred[0]
                
                # Extract edges belonging to first graph
                E0_mask = (batch.edge_index[0] < N) & (batch.edge_index[1] < N)
                edge_index_single = batch.edge_index[:, E0_mask]

                visualize_shapes(
                    pos_gt=pos_gt_single.cpu(),
                    pos_init=pos_init_single.cpu(),
                    pos_pred=pos_pred_single.cpu(),
                    edge_index=edge_index_single.cpu(),
                    title="Example 0: Initialization vs Prediction",
                )

    # ----------------------------
    # Final P-MPJPE
    # ----------------------------
    all_errors = torch.cat(all_errors)
    print(f"\nMean P-MPJPE: {all_errors.mean().item():.4f} mm")


if __name__ == "__main__":
    test()
