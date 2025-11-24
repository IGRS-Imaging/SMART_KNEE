import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import csv

import config
from data.dataset import LoadFemurDataset
from models import LandmarkCompletionModel
from losses import CompositeLoss

def log_to_csv(epoch, avg_loss, avg_pos, avg_edge, log_path):
    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'avg_loss', 'avg_pos', 'avg_edge']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerow({'epoch': epoch, 'avg_loss': avg_loss, 'avg_pos': avg_pos, 'avg_edge': avg_edge})

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure augmentation is ON
    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV, augment=True)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = LandmarkCompletionModel().to(device)
    
    criterion = CompositeLoss(w_pos=config.W_POS, w_edge=config.W_EDGE, w_angle=config.W_ANGLE).to(device)
    
    # Add weight_decay to prevent weights from growing too large (Stability)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    
    # Scheduler: Reduce LR if loss stops improving for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    best_loss = float('inf')
    
    if os.path.exists(config.LOG_PATH): os.remove(config.LOG_PATH)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        loss_log = {"pos": 0.0, "edge": 0.0}

        pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)
        for batch in pbar:
            batch = batch.to(device)

            pred, _ = model(batch)

            B = batch.num_graphs
            N = config.NUM_NODES
            
            loss, info = criterion(
                pred=pred.view(B, N, 3), 
                target=batch.pos.view(B, N, 3), 
                known_mask=batch.known_mask.view(B, N),
                edge_index=batch.edge_index,
                edge_attr_gt=batch.edge_attr
            )

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping: Prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item() * B
            loss_log["pos"] += info["L_pos"] * B
            loss_log["edge"] += info["L_edge"] * B
            pbar.set_postfix({"L_pos": f"{info['L_pos']:.2f}"})

        avg_loss = total_loss / len(dataset)
        avg_pos = loss_log['pos'] / len(dataset)
        avg_edge = loss_log['edge'] / len(dataset)
        
        # Step the scheduler
        scheduler.step(avg_loss)
        
        print(f"Ep {epoch}: Avg {avg_loss:.4f} [Pos: {avg_pos:.4f}, Edge: {avg_edge:.4f}]")
        log_to_csv(epoch, avg_loss, avg_pos, avg_edge, config.LOG_PATH)

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

if __name__ == "__main__":
    train()