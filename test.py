# test.py
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import config
from data.dataset import LoadFemurDataset
from models import LandmarkCompletionModel
from losses import procrustes_align
from utils.visualization import visualize_shapes 

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on device:", device)

    dataset = LoadFemurDataset(config.LANDMARKS_CSV, config.EDGES_CSV, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = LandmarkCompletionModel().to(device)
    
    try:
        model.load_state_dict(torch.load(config.TEST_MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Checkpoint not found. Please train the model first.")
        return

    model.eval()
    errors = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = batch.to(device)
            current_subject = batch.subject[0] if isinstance(batch.subject, (list, tuple)) else str(batch.subject)

            # 1. Forward pass 
            # The model now returns coordinates rigidly aligned to the input known nodes
            # No extra scaling/unscaling needed here if the model does it internally.
            pred_mm, pos_init_mm = model(batch) 

            # Reshape (B=1, N, 3)
            pred_reshaped = pred_mm.view(1, config.NUM_NODES, 3)
            target_reshaped = batch.pos.view(1, config.NUM_NODES, 3)
            known_mask = batch.known_mask.view(1, config.NUM_NODES)
            
            # 2. Error Calculation (P-MPJPE)
            # Since the model output is already anchored, we can check raw error
            # OR we can still use Procrustes to check purely shape fidelity.
            # Standard protocol usually allows Procrustes alignment for error metric:
            pred_aligned, _, _ = procrustes_align(pred_reshaped, target_reshaped)
            
            unknown_mask = (~known_mask).float().unsqueeze(-1)
            diff = (pred_aligned - target_reshaped) * unknown_mask
            
            # Mean Euclidean distance over UNKNOWN nodes only
            err = diff.norm(dim=-1).sum() / unknown_mask.sum()
            errors.append(err.item())

            # 3. Visualization (idx == 150 or any specific index)
            if idx == 222:
                print(f"Visualizing Sample {idx} | Subject: {current_subject} | Error: {err.item():.3f} mm")
                
                # Visualize the OUTPUT of the model directly (pred_mm) 
                # to verify the orientation fix works.
                visualize_shapes(
                    pos_gt=target_reshaped[0].cpu(),      
                    pos_init=pos_init_mm.cpu(),        
                    pos_pred=pred_mm.cpu(), # Visualizing RAW model output to check orientation
                    edge_index=batch.edge_index.cpu(),
                    title=f"Sample {idx}: Raw Output vs GT (Error: {err.item():.3f}mm)"
                )
    
    mean_error = np.mean(errors)
    print(f"\n✅ Overall Mean P-MPJPE (Unknown Nodes): {mean_error:.3f} mm")

if __name__ == "__main__":
    test()