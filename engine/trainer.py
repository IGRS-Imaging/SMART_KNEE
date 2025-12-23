import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import config
from losses.composite_loss import CompositeLoss
from models import LandmarkCompletionModel

class EarlyStopping:
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
    
    model = LandmarkCompletionModel().to(device)
    
    if args.resume and os.path.exists(config.CHECKPOINT_PATH):
        print(f"Resuming from {config.CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH))

    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True
    )
    
    # Init Loss with new W_GLOBAL
    criterion = CompositeLoss(
        w_pos=config.W_POS, 
        w_edge=config.W_EDGE, 
        w_angle=config.W_ANGLE,
        w_global=config.W_GLOBAL
    )
    early_stopper = EarlyStopping(patience=30) 

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    print(f"Starting training on {device} for {config.NUM_EPOCHS} epochs.")
    print(f"Structure Weights -> Angle: {config.W_ANGLE}, Global: {config.W_GLOBAL}")
    print("-" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # --- TRAIN LOOP ---
        model.train()
        running_loss = 0.0
        
        # Track components for logging
        loss_components = {"L_pos": 0, "L_edge": 0, "L_angle": 0, "L_global": 0}
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_mm, _ = model(batch)
            
            # Reshape
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
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
            # Aggregate components
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
                pred_mm, _ = model(batch)
                
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
            # Avg components per batch
            n_batches = len(train_loader)
            log_str = (f"Ep {epoch} | Pos: {loss_components['L_pos']/n_batches:.2f} "
                       f"Edge: {loss_components['L_edge']/n_batches:.2f} "
                       f"Ang: {loss_components['L_angle']/n_batches:.4f} "
                       f"Glob: {loss_components['L_global']/n_batches:.2f}")
            print(log_str)

        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f}{save_msg}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
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
    print(f"\nTraining Complete. Curve at {plot_path}")