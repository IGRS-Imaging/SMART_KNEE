# train.py
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os

import config
from data.dataset import LoadFemurDataset
from models import LandmarkCompletionModel
# Ensure this matches the filename where you saved the new class
from losses import CompositeLoss1

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)

    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV, augment=False)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = LandmarkCompletionModel().to(device)
    
    # --- FIX: Initialize with the NEW arguments matching the Class Definition ---
    criterion = CompositeLoss1(
        w_proc=config.W_PROC, 
        w_icp=config.W_ICP, 
        w_tps=config.W_TPS, 
        w_edge=config.W_EDGE
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    best_loss = float('inf')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        
        # --- FIX: Update Log dictionary to track new losses ---
        loss_log = {"proc": 0.0, "icp": 0.0, "tps": 0.0, "edge": 0.0}

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", ncols=120)
        for batch in pbar:
            batch = batch.to(device)

            pred, pos_init = model(batch)

            B = batch.num_graphs
            N = config.NUM_NODES
            
            pred_reshaped = pred.view(B, N, 3)
            target_reshaped = batch.pos.view(B, N, 3)
            known_reshaped = batch.known_mask.view(B, N)

            loss, info = criterion(
                pred=pred_reshaped, 
                target=target_reshaped, 
                known_mask=known_reshaped,
                edge_index=batch.edge_index,
                edge_attr_gt=batch.edge_attr
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            
            # --- FIX: Log the new keys ---
            loss_log["proc"] += info["L_proc"] * B
            loss_log["icp"]  += info["L_icp"] * B
            loss_log["tps"]  += info["L_tps"] * B
            loss_log["edge"] += info["L_edge"] * B

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_loss = total_loss / len(dataset)
        
        # --- FIX: Update Print Statement ---
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f} "
              f"[Proc: {loss_log['proc']/len(dataset):.3f}, "
              f"ICP: {loss_log['icp']/len(dataset):.3f}, "
              f"TPS: {loss_log['tps']/len(dataset):.3f}, "
              f"Edge: {loss_log['edge']/len(dataset):.3f}]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(">>> New best model saved.")

if __name__ == "__main__":
    train()