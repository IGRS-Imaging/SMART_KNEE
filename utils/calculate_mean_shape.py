# util/calculate_mean_shape.py
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def extract_coords(row):
    coords = row.iloc[1: 1 + 3 * config.NUM_NODES].astype(float).values
    coords = coords.reshape(config.NUM_NODES, 3)
    return coords

def calculate_mean_shape():
    print(f"Calculating Mean Canonical Shape from: {config.LANDMARKS_CSV}")
    
    df = pd.read_csv(config.LANDMARKS_CSV, skiprows=[1])
    
    canonical_shapes = []
    count_used = 0
    
    # 1. STRICT FILTER: Only use explicit Right Femurs
    for idx, row in df.iterrows():
        subject = str(row["Source"]).upper()
        
        # We ONLY trust files labeled _R for the template
        if subject.endswith("_R"):
            coords = extract_coords(row)
            
            # Center this shape
            centroid = np.mean(coords, axis=0)
            canonical_shapes.append(coords - centroid)
            count_used += 1

    if count_used == 0:
        print("ERROR: No subjects ending in '_R' found!")
        return

    canonical_shapes = np.array(canonical_shapes)
    
    # 2. Calculate Mean
    mean_canonical_shape = np.mean(canonical_shapes, axis=0)
    
    # 3. Save
    save_path = config.MEAN_SHAPE_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, mean_canonical_shape)
    
    print(f"✅ Mean Canonical Shape saved to: {save_path}")
    print(f"   Based on {count_used} samples (Right side only).")

if __name__ == "__main__":
    calculate_mean_shape()