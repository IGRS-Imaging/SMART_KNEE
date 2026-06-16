import argparse
import os
import torch
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import config
from data.dataset import get_dataloaders
from engine.trainer import train_engine
from engine.evaluator import evaluate_model
from models import LandmarkCompletionModel
from utils.calculate_mean_shape import calculate_mean_shape
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Femur/Tibia Landmark Completion")
    
    # Bone Selection
    parser.add_argument('--bone', type=str, default='femur', choices=['femur', 'tibia'], 
                        help='Bone type: femur or tibia')

    # Modes
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'mean_shape'], 
                        help='Mode: train, test or mean_shape')
    
    # Paths (Override config if provided)
    parser.add_argument('--csv_path', type=str, default=None, help='Override path to landmarks CSV')
    parser.add_argument('--edges_path', type=str, default=None, help='Override path to edges CSV')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Override path to save/load model')
    
    # Training Params
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    # Output Options
    parser.add_argument('--save_csv', action='store_true', 
                        help='If True (in test mode), saves predictions to CSV for pipeline.')

    return parser.parse_args()

def save_edges_to_csv(prediction_results, template_path, output_path, bone_type):
    """
    Calculates edge lengths from predicted landmarks and saves in the Edge CSV format.
    """
    print(f"   -> Generating edge distances for {os.path.basename(output_path)}...")
    
    try:
        # Load the template to get the Edge definitions (V1, V2)
        df_template = pd.read_csv(template_path)
    except FileNotFoundError:
        print(f"⚠️ Warning: Edge template not found at {template_path}. Skipping edge saving.")
        return

    # 1. Prepare Output DataFrame (Copy EdgeID, V1, V2)
    # This preserves the original structure
    out_df = df_template[['EdgeID', 'V1', 'V2']].copy()
    
    # 2. Get 0-based indices for calculation (Input CSV is 1-based)
    v1_idx = out_df['V1'].astype(int).values - 1
    v2_idx = out_df['V2'].astype(int).values - 1
    
    # 3. Determine number of nodes
    num_nodes = 12 if bone_type.lower() == 'femur' else 11
    
    # 4. Iterate over all predicted subjects
    # prediction_results structure: [Source, x0, y0, z0, ..., Metrics...]
    for row in prediction_results:
        subj_id = row[0]
        
        # Extract only the coordinates (skip Source and trailing Metrics)
        # Coordinates correspond to index 1 to (1 + 3*N)
        coords_flat = row[1 : 1 + 3*num_nodes]
        
        # Reshape to (N, 3) for vectorized calculation
        coords = np.array(coords_flat).reshape(num_nodes, 3)
        
        # Vectorized Euclidean Distance Calculation
        p1 = coords[v1_idx]
        p2 = coords[v2_idx]
        dists = np.linalg.norm(p1 - p2, axis=1)
        
        # Add column for this subject
        out_df[str(subj_id)] = dists
        
    # Save
    out_df.to_csv(output_path, index=False)
    print(f"   ✅ Saved edge predictions to {output_path}")

def save_predictions_to_csv(model, loader, device, output_path, bone_type):
    """
    Saves predictions in the EXACT format of the input CSVs (Double Header).
    Then automatically calculates and saves the corresponding Edge CSV.
    """
    model.eval()
    results = []
    
    # --- 1. Define Headers based on Bone Type ---
    if bone_type.lower() == 'femur':
        # Row 1: Landmark Names + Metric Headers
        header_row_1 = [
            'Source', 
            'HIP CENTRE ', '', '', 
            'FEMUR KNEE CENTRE ', '', '', 
            'MEDIAL EPICONDYLE', '', '', 
            'LATERAL EPICONDYLE', '', '', 
            'MEDIAL DISTAL CONDYLE', '', '', 
            'LATERAL DISTAL CONDYLE', '', '', 
            'MEDIAL POSTERIOR CONDYLE', '', '', 
            'LATERAL POSTERIOR CONDYLE', '', '', 
            'Medial Anterior Cortex', '', '', 
            'Lateral Anterior Cortex', '', '', 
            'Medial Posterior Proximal', '', '', 
            'Lateral Posterior Proximal', '', '', 
            'MA Length', 'TEA Length', 'AP Length'
        ]
        num_metrics = 3
        
    elif bone_type.lower() == 'tibia':
        header_row_1 = [
            'Source',
            'TIBIAL KNEE CENTRE ', '', '',
            'MEDIAL PLATEAU', '', '',
            'LATERAL PLATEAU', '', '',
            'MEDIAL MALLEOLUS', '', '',
            'LATERAL MALLEOLUS', '', '',
            'TUBEROSITY', '', '',
            'PCL', '', '',
            'ANKLE ', '', '',
            'PCL OPP', '', '',
            'Medial epicondyle', '', '',
            'Lateral epicondyle', '', '',
            'Plateau_Length', 'Knee_Centre_To_Ankle_Length'
        ]
        num_metrics = 2
    else:
        raise ValueError(f"Unknown bone type: {bone_type}")

    # Row 2: Units / Coordinate Labels
    num_nodes = config.NUM_NODES
    header_row_2 = [''] + ['X', 'Y', 'Z'] * num_nodes + [''] * num_metrics

    print(f"Generating predictions for {output_path}...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting"):
            batch = batch.to(device)
            
            # Forward pass
            pred_mm, _, _ = model(batch)
            
            B = batch.num_graphs
            N = config.NUM_NODES
            pred = pred_mm.view(B, N, 3).cpu().numpy()
            
            for b in range(B):
                # 1. Get Subject ID
                if hasattr(batch, 'subject'):
                    if isinstance(batch.subject, list):
                        subj_id = batch.subject[b]
                    else:
                        subj_id = batch.subject[b] if B > 1 else batch.subject
                else:
                    subj_id = f"unknown"

                # 2. Flatten coordinates
                flat_coords = pred[b].flatten().tolist()
                
                # 3. Create Row [Source] + [Coords...] + [NaNs for Metrics]
                metrics_placeholder = [np.nan] * num_metrics
                row = [subj_id] + flat_coords + metrics_placeholder
                results.append(row)

    # --- 2. Write Landmark CSV (Double Header) ---
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_row_1)
        writer.writerow(header_row_2)
        writer.writerows(results)
    print(f"✅ Saved {len(results)} landmark predictions to {output_path}")

    # --- 3. Save Corresponding Edge CSV ---
    # Create name: pred_test_femur.csv -> edges_pred_test_femur.csv
    base_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    edge_output_path = os.path.join(base_dir, f"edges_{base_name}")
    
    # Use the config.EDGES_CSV as the template for V1/V2
    save_edges_to_csv(results, config.EDGES_CSV, edge_output_path, bone_type)

def main():
    args = parse_args()
    
    # 1. SETUP CONFIGURATION
    config.set_bone_config(args.bone)

    if args.mode == 'mean_shape':
        print(f"Generating mean shape for {args.bone}..")
        calculate_mean_shape(args.bone)
        return
    
    if args.csv_path: config.LANDMARKS_CSV = args.csv_path
    if args.edges_path: config.EDGES_CSV = args.edges_path
    if args.checkpoint_path: config.CHECKPOINT_PATH = args.checkpoint_path
    
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    # 2. LOAD DATA
    print(f"Loading data from {config.LANDMARKS_CSV}...")
    print(f"Bone Type: {args.bone.upper()} | Nodes: {config.NUM_NODES}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        config.LANDMARKS_CSV, 
        config.EDGES_CSV, 
        args.batch_size
    )
    print(f"Data Split -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. EXECUTION MODES
    if args.mode == 'train':
        print("\nStarting Training Mode...")
        train_engine(train_loader, val_loader, args)
        
    elif args.mode == 'test':
        print("\nStarting Testing Mode...")
        if not os.path.exists(config.CHECKPOINT_PATH):
            print(f"Error: No checkpoint found at {config.CHECKPOINT_PATH}")
            return

        model = LandmarkCompletionModel().to(device)
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
        print(f"Model loaded from {config.CHECKPOINT_PATH}")
        
        # A. Standard Evaluation
        evaluate_model(model, test_loader, device, return_detailed=True, verbose=True)

        # B. Save Predictions for Pipeline
        if args.save_csv:
            print("\n--- Saving Predictions for Downstream Pipeline ---")
            
            # This generates both predictions.csv AND edges_predictions.csv
            
            # Save Test Set
            test_out = os.path.join(config.PRED_OUTPUT_DIR, f"pred_test_{args.bone}.csv")
            save_predictions_to_csv(model, test_loader, device, test_out, args.bone)
            
            # Save Train Set
            train_out = os.path.join(config.PRED_OUTPUT_DIR, f"pred_train_{args.bone}.csv")
            save_predictions_to_csv(model, train_loader, device, train_out, args.bone)

if __name__ == "__main__":
    main()