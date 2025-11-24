import torch
import os
import csv
from tqdm import tqdm
from models import LandmarkCompletionModel
from losses import CompositeLoss
import config
from .evaluator import evaluate_model

def log_to_csv(epoch, train_loss, val_error, log_path):
    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_error_mm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerow({'epoch': epoch, 'train_loss': train_loss, 'val_error_mm': val_error})

def train_engine(train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = LandmarkCompletionModel().to(device)
    
    # Load checkpoint if exists and requested
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"Resuming from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    criterion = CompositeLoss(w_pos=config.W_POS, w_edge=config.W_EDGE, w_angle=config.W_ANGLE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_error = float('inf')
    
    # Clear log if starting fresh
    if not args.resume and os.path.exists(config.LOG_PATH):
        os.remove(config.LOG_PATH)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # --- TRAIN LOOP ---
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}", ncols=100)
        for batch in pbar:
            batch = batch.to(device)
            pred, _ = model(batch)
            
            B, N = batch.num_graphs, config.NUM_NODES
            
            loss, info = criterion(
                pred=pred.view(B, N, 3), 
                target=batch.pos.view(B, N, 3), 
                known_mask=batch.known_mask.view(B, N),
                edge_index=batch.edge_index,
                edge_attr_gt=batch.edge_attr
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * B
            pbar.set_postfix({"Loss": f"{loss.item():.2f}"})

        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # --- VALIDATION LOOP ---
        # We use the evaluator to get real MM error
        if epoch % 1 == 0: # Validate every epoch
            val_error = evaluate_model(model, val_loader, device)
            
            scheduler.step(val_error)
            log_to_csv(epoch, avg_train_loss, val_error, config.LOG_PATH)

            print(f"Ep {epoch}: Train Loss {avg_train_loss:.4f} | Val Err {val_error:.4f} mm")

            if val_error < best_val_error:
                best_val_error = val_error
                os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), args.checkpoint_path)
                print(f"--> Best model saved!")