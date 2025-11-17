# train.py

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config
from data import LoadFemurDataset
from models import LandmarkCompletionModel
from losses import CompositeLoss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = LandmarkCompletionModel().to(device)
    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", ncols=100)

        for batch in progress:
            batch = batch.to(device)

            # --- Model forward ---
           # --- Model forward ---
            pred_flat, pos_init_flat = model(batch)

            B = batch.num_graphs
            N = config.NUM_NODES

            # reshape both outputs
            pred = pred_flat.view(B, N, 3)
            pos_init = pos_init_flat.view(B, N, 3)

            # Correct ground-truth
            target = batch.pos.view(B, N, 3)

            loss, info = criterion(pred, target)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            progress.set_postfix(
                {"loss": f"{loss.item():.4f}", "L_align": f"{info['L_align']:.4f}", "L_shape": f"{info['L_shape']:.4f}"}
            )

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

        # Save best so far (simple version: overwrite)
        torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    print("Training finished. Saved model to", config.CHECKPOINT_PATH)


if __name__ == "__main__":
    train()
