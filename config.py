import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data Settings
NUM_NODES = 12
KNOWN_IDS = [0, 2, 7]   

# Model Architecture
FEAT_DIM = 128
HIDDEN_DIM = 256
GNN_LAYERS = 4
EDGE_DIM = 1

# Training
BATCH_SIZE = 16
LR = 1e-4         
NUM_EPOCHS = 150

# Normalization & Template
GLOBAL_SCALE = 500.0 

MEAN_SHAPE_PATH = os.path.join(PROJECT_ROOT, "data", "mean_canonical_shape.npy") 
LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "FEMUR_LANDMARKS.csv")
EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "Femur_Edges.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
TEST_MODEL_PATH = CHECKPOINT_PATH
LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "training_log.csv") 

# Loss Weights
# Increased Pos weight to fight for the last mm
W_POS =2.0     
W_EDGE = 1.0    
W_GLOBAL =1.0  
W_ANGLE = 0.5