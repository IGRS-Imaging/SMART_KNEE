#engine/trainer.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import config
from losses.composite_loss import CompositeLoss
from models import LandmarkCompletionModel

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_engine(train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model & Loss
    model = LandmarkCompletionModel().to(device)
    
    if args.resume and os.path.exists(config.CHECKPOINT_PATH):
        print(f"Resuming from {config.CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH))

    # Using AdamW for better weight decay handling (helps generalization)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    
    # Scheduler: Reduce LR if validation loss plateaus (Stabilization)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    criterion = CompositeLoss(w_pos=config.W_POS, w_edge=config.W_EDGE, w_angle=config.W_ANGLE)
    early_stopper = EarlyStopping(patience=30) # Stop if no improvement for 30 epochs

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    print(f"Starting training on {device} for {config.NUM_EPOCHS} epochs.")
    print(f"Output path: {config.CHECKPOINT_PATH}")
    print("-" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # --- TRAIN LOOP ---
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            pred_mm, pos_init_mm = model(batch)
            
            # Reshape for Loss
            B = batch.num_graphs
            N = config.NUM_NODES
            pred = pred_mm.view(B, N, 3)
            target = batch.pos.view(B, N, 3)
            known_mask = batch.known_mask.view(B, N)
            
            # Loss Calculation
            loss, _ = criterion(
                pred=pred, 
                target=target, 
                known_mask=known_mask, 
                edge_index=batch.edge_index, 
                edge_attr_gt=batch.edge_attr
            )
            
            loss.backward()
            
            # Gradient Clipping (CRITICAL for stability)
            # Prevents gradients from exploding and causing "way off" predictions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- VALIDATION LOOP (Calculate Loss, not just Metric) ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_mm, _ = model(batch)
                
                # Reshape
                B = batch.num_graphs
                N = config.NUM_NODES
                pred = pred_mm.view(B, N, 3)
                target = batch.pos.view(B, N, 3)
                known_mask = batch.known_mask.view(B, N)
                
                val_loss_item, _ = criterion(
                    pred=pred, 
                    target=target, 
                    known_mask=known_mask, 
                    edge_index=batch.edge_index, 
                    edge_attr_gt=batch.edge_attr
                )
                running_val_loss += val_loss_item.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # --- CHECKPOINTING & LOGGING ---
        # Update Scheduler
        scheduler.step(avg_val_loss)
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            save_msg = " [Saved]"
        else:
            save_msg = ""

        # Concise Print
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}{save_msg}")

        # Early Stopping check
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered. Training finished.")
            break

    # --- PLOTTING ---
    plot_path = os.path.join(os.path.dirname(config.CHECKPOINT_PATH), "loss_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"\nTraining Complete. Loss curve saved to {plot_path}")