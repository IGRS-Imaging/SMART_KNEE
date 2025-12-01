import os
import random
import numpy as np
import torch
from pathlib import Path

ROOT_DIR = r"C:\Users\sweth\Downloads\VAE_Data_Prep\WITH_SOURCE_AUGMENTED_FEMUR"
LANDMARKS_CSV = r"C:\Users\sweth\Downloads\VAE_Data_Prep\Sheets_Augmented_femur+tibia\Femur_Landmarks_Augmented.csv"
EDGES_CSV = r"C:\Users\sweth\Downloads\VAE_Data_Prep\Sheets_Augmented_femur+tibia\Femur_Edges.csv"
CHECKPOINT_DIR = "checkpoints"
VIS_DIR = "vis"
NUM_POINTS = 4096
BATCH_SIZE = 10
EPOCHS = 50
# LR = 1e-4
LR = 5e-5
LATENT_DIM = 64
COND_EMB_DIM = 256
IN_DIM_COND = 42
VAL_SPLIT = 0.15 
RANDOM_SEED = 42
KL_WEIGHT_BASE = 0.005
NUM_WORKERS = 2

# Setup directories and seeds
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"Config loaded: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, Device={DEVICE}")
if __name__ == "__main__":
    import Dataloader   # runs next file