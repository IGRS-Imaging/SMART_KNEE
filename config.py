import os

# Get the absolute path of the directory where config.py 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data Settings
NUM_NODES = 12
KNOWN_IDS = [0, 2, 7, 9]   

# Model Architecture
FEAT_DIM = 128
HIDDEN_DIM = 128
GNN_LAYERS = 5
EDGE_DIM = 1      

# Training
BATCH_SIZE = 8
LR = 1e-4
NUM_EPOCHS = 200

# Normalization & Template
GLOBAL_SCALE = 500.0 

MEAN_SHAPE_PATH = os.path.join(PROJECT_ROOT, "data", "mean_canonical_shape.npy") 
LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "Femur_Landmarks.csv")
EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "Edges.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
TEST_MODEL_PATH = CHECKPOINT_PATH
LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "training_log.csv") 

# Loss Weights
W_POS = 10.0      
W_EDGE = 2.0