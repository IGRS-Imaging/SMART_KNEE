import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import config
from losses.composite_loss import CompositeLoss
from models import LandmarkCompletionModel

class EarlyStopping:
    def __init__(self, patience=40, min_delta=0.001):
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
    
    model = LandmarkCompletionModel().to(device)
    
    if args.resume and os.path.exists(config.CHECKPOINT_PATH):
        print(f"Resuming from {config.CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH))

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LR, 
        weight_decay=5e-4,  # Moderate regularization
        betas=(0.9, 0.999)
    )
    
    # Use ReduceLROnPlateau - more stable than cosine annealing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=20,
        verbose=True,
        min_lr=1e-6
    )
    
    criterion = CompositeLoss(
        w_pos=config.W_POS, 
        w_edge=config.W_EDGE, 
        w_angle=config.W_ANGLE,
        w_local_struct=config.W_LOCAL_STRUCT
    )
    early_stopper = EarlyStopping(patience=50)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    print(f"Starting training on {device} for {config.NUM_EPOCHS} epochs.")
    print(f"Model: FEAT={config.FEAT_DIM}, HIDDEN={config.HIDDEN_DIM}, LAYERS={config.GNN_LAYERS}")
    print(f"Weights -> Pos:{config.W_POS}, Edge:{config.W_EDGE}, Angle:{config.W_ANGLE}, Struct:{config.W_LOCAL_STRUCT}")
    print("-" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # --- TRAIN LOOP ---
        model.train()
        running_loss = 0.0
        
        loss_components = {"L_pos": 0, "L_edge": 0, "L_angle": 0, "L_local_struct": 0}
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_mm, _, confidence = model(batch)
            
            B = batch.num_graphs
            N = config.NUM_NODES
            pred = pred_mm.view(B, N, 3)
            target = batch.pos.view(B, N, 3)
            known_mask = batch.known_mask.view(B, N)
            
            loss, comp = criterion(
                pred=pred, 
                target=target, 
                known_mask=known_mask, 
                edge_index=batch.edge_index, 
                edge_attr_gt=batch.edge_attr
            )
            
            # Check for NaN
            if torch.isnan(loss):
                print("WARNING: NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
            for k, v in comp.items():
                loss_components[k] += v

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- VALIDATION LOOP ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_mm, _, _ = model(batch)
                
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

        # --- CHECKPOINTING ---
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            save_msg = " [Saved]"
        else:
            save_msg = ""

        # Verbose log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            n_batches = len(train_loader)
            log_str = (f"Ep {epoch} | Pos: {loss_components['L_pos']/n_batches:.2f} "
                       f"Edge: {loss_components['L_edge']/n_batches:.2f} "
                       f"Ang: {loss_components['L_angle']/n_batches:.4f} "
                       f"Struct: {loss_components['L_local_struct']/n_batches:.2f}")
            print(log_str)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.2e}{save_msg}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # --- PLOTTING ---
    plot_path = os.path.join(os.path.dirname(config.CHECKPOINT_PATH), "loss_curve_v3.png")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (v3 - Stable)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"\nTraining Complete. Curve at {plot_path}")
    print(f"Best Val Loss: {best_val_loss:.4f}")