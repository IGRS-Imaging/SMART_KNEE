# Data
NUM_NODES = 12
# Example indices: hip centre, medial epicondyle, lateral proximal condyle
KNOWN_IDS = [0, 2, 7]   

# Model Architecture
FEAT_DIM = 128
HIDDEN_DIM = 128
GNN_LAYERS = 4
EDGE_DIM = 1      # Changed from 0 to 1 (Euclidean distance is the feature)

# Training Hyperparameters
BATCH_SIZE = 8
LR = 1e-3
NUM_EPOCHS = 200

# Loss Weights
W_ALIGN = 1.0     # Procrustes alignment weight
W_SHAPE = 0.1     # Pairwise shape consistency weight
W_PROC = 0.2      # Rigid Alignment weight
W_ICP  = 0.5      # Chamfer/Nearest Neighbor weight
W_TPS  = 1.0      # Non-rigid warping weight
W_POS = 1.0      # Main position MSE
W_EDGE = 1.0     # Edge consistency

# Paths
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
TEST_MODEL_PATH = CHECKPOINT_PATH

LANDMARKS_CSV = "./data/Femur_Landmarks.csv"
EDGES_CSV = "./data/Edges.csv"