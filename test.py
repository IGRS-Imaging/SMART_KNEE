# test.py
import torch
import numpy as np
import matplotlib.pyplot as plt # Import for the final P-MPJPE plot
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

    # Get scale factors from the dataset to properly un-normalize
    # We will assume that the DataLoader/Dataset is modified to include these, 
    # OR that you stick to the global scale factor if the data is normalized relative to a global mean.
    # For robust testing, the scale factors should ideally come from the batch itself if per-sample normalized.
    # Since your current data normalization is PER-FEMUR, we must rely on the GLOBAL_SCALE factor provided in config.
    scale_factor = getattr(config, 'GLOBAL_SCALE', 1.0) 

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = batch.to(device)
            # Ensure subject is a string
            current_subject = batch.subject[0] if isinstance(batch.subject, (list, tuple)) else str(batch.subject)

            # 1. Forward pass (Normalized outputs)
            # pred_norm is the final output (with known points hard-frozen)
            pred_norm, pos_init_norm = model(batch) 

            # 2. Un-Scale back to Millimeters for visualization/metric
            # NOTE: This assumes 'batch.pos' (GT) is also the normalized coordinates 
            # and that 'scale_factor' represents the average mm/normalized-unit ratio.
            pred_mm = pred_norm * scale_factor
            target_mm = batch.pos * scale_factor
            pos_init_mm = pos_init_norm * scale_factor

            # Reshape (B=1, N, 3)
            pred_reshaped = pred_mm.view(1, config.NUM_NODES, 3)
            target_reshaped = target_mm.view(1, config.NUM_NODES, 3)
            known_mask = batch.known_mask.view(1, config.NUM_NODES)

            # 3. Rigid Alignment (Procrustes)
            # We align the PREDICTION (pred_reshaped) to the TARGET (target_reshaped)
            pred_aligned, _, _ = procrustes_align(pred_reshaped, target_reshaped)

            # 4. Error Calculation (P-MPJPE on Unknown nodes only)
            # The error is calculated on the Procrustes-aligned prediction
            unknown_mask = (~known_mask).float().unsqueeze(-1)
            diff = (pred_aligned - target_reshaped) * unknown_mask
            
            # The P-MPJPE formula: Mean Euclidean distance over UNKNOWN nodes only
            err = diff.norm(dim=-1).sum() / unknown_mask.sum()
            errors.append(err.item())

            # 5. Visualization Call
            # Visualize the sample that was requested (idx=200)
            if idx == 150:
                print(f"Visualizing Sample {idx} | Subject: {current_subject} | P-MPJPE: {err.item():.3f} mm")
                
                # Reshape init for consistent visualization
                pos_init_reshaped = pos_init_mm.view(config.NUM_NODES, 3)
                
                # Use the Procrustes-Aligned prediction for the 'Pred' plot
                visualize_shapes(
                    pos_gt=target_reshaped[0].cpu(),      
                    pos_init=pos_init_reshaped.cpu(),        
                    pos_pred=pred_aligned[0].cpu(),       
                    edge_index=batch.edge_index.cpu(),
                    title=f"Prediction Sample {idx} (P-MPJPE: {err.item():.3f}mm)"
                )
                
                # break # Remove 'break' if you want to calculate the full mean error
                
    
    mean_error = np.mean(errors)
    print(f"\n✅ Overall Mean P-MPJPE (Unknown Nodes): {mean_error:.3f} mm")

if __name__ == "__main__":
    # NOTE: You must have 'GLOBAL_SCALE' defined in config.py for the 
    # error to be interpreted in 'mm'. If not, it defaults to 1.0 (normalized units).
    test()