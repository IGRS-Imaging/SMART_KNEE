# train.py
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config
from data.dataset import LoadFemurDataset
from models import LandmarkCompletionModel
from losses import CompositeLoss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV, augment=True)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = LandmarkCompletionModel().to(device)
    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", ncols=120)
        for batch in pbar:
            batch = batch.to(device)

            pred, pos_init = model(batch)

            B = batch.num_graphs
            N = config.NUM_NODES
            pred = pred.view(B, N, 3)
            target = batch.pos.view(B, N, 3)

            known = batch.known_mask.view(B, N)
            loss, info = criterion(pred, target, known)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        print(f"Epoch {epoch}: Avg Loss = {total_loss/len(dataset):.4f}")

        torch.save(model.state_dict(), config.CHECKPOINT_PATH)


if __name__ == "__main__":
    train()
