# train.py
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os

import config
from data.dataset import LoadFemurDataset
from models import LandmarkCompletionModel
from losses import CompositeLoss # Now assumes the updated one

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Dataset
    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV, augment=True)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = LandmarkCompletionModel().to(device)
    
    # NEW: Use simpler weights
    criterion = CompositeLoss(w_pos=config.W_POS, w_edge=config.W_EDGE).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    best_loss = float('inf')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        loss_log = {"pos": 0.0, "edge": 0.0}

        pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)
        for batch in pbar:
            batch = batch.to(device)

            # Forward pass (now returns rigid-anchored predictions)
            pred, _ = model(batch)

            B = batch.num_graphs
            N = config.NUM_NODES
            
            pred_reshaped = pred.view(B, N, 3)
            target_reshaped = batch.pos.view(B, N, 3)
            known_reshaped = batch.known_mask.view(B, N)

            # Calculate Loss (Direct MSE vs GT)
            loss, info = criterion(
                pred=pred_reshaped, 
                target=target_reshaped, 
                known_mask=known_reshaped,
                edge_index=batch.edge_index,
                edge_attr_gt=batch.edge_attr
            )

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping helps stabilize EGNN coords
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            total_loss += loss.item() * B
            loss_log["pos"] += info["L_pos"] * B
            loss_log["edge"] += info["L_edge"] * B

            pbar.set_postfix({"L_pos": f"{info['L_pos']:.2f}"})

        avg_loss = total_loss / len(dataset)
        avg_pos = loss_log['pos'] / len(dataset)
        avg_edge = loss_log['edge'] / len(dataset)
        
        print(f"Ep {epoch}: Avg {avg_loss:.4f} [Pos: {avg_pos:.4f}, Edge: {avg_edge:.4f}]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

if __name__ == "__main__":
    train()