#util/calculate_mean_shape.py
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import config and helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset import extract_coords, get_chirality # Import necessary helpers

def calculate_mean_shape():
    print(f"Calculating Mean Canonical Shape from: {config.LANDMARKS_CSV}")
    
    # 1. Read CSV
    try:
        # Assuming extract_coords needs the skiprows=[1] handling, consistent with dataset.py
        df = pd.read_csv(config.LANDMARKS_CSV, skiprows=[1]) 
    except FileNotFoundError:
        print(f"ERROR: Could not find {config.LANDMARKS_CSV}")
        return

    # Define the reflection matrix (e.g., flips X-axis)
    # This reflects Left into the Canonical (Right) space.
    # If the coordinate system is X=Lateral/Medial, Y=Proximal/Distal, Z=Anterior/Posterior,
    # then flipping X is the correct reflection across the sagittal plane.
    reflection_matrix = np.diag([-1, 1, 1]) 
    
    canonical_shapes = []
    
    # 2. Extract and Normalize Shapes
    for idx, row in df.iterrows():
        # Get coordinates
        coords = extract_coords(row) 
        
        subject = row["Source"]
        # get_chirality returns 0 for Left, 1 for Right
        side = get_chirality(subject) 
        
        # --- CHIRALITY NORMALIZATION ---
        if side == 0: # If Left Femur
            # Reflect the shape into the canonical (Right) space
            coords = coords @ reflection_matrix 
            
        # Center this shape to origin
        centroid = np.mean(coords, axis=0)
        canonical_shapes.append(coords - centroid)
        
    canonical_shapes = np.array(canonical_shapes)
    
    # 3. Calculate Mean Canonical Shape
    mean_canonical_shape = np.mean(canonical_shapes, axis=0)
    
    # 4. Save
    save_path = config.MEAN_SHAPE_PATH
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.save(save_path, mean_canonical_shape)
    print(f"✅ Mean canonical shape saved to: {save_path}")
    print("Shape dimensions:", mean_canonical_shape.shape)

if __name__ == "__main__":
    calculate_mean_shape()