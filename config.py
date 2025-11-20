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
W_EDGE  = 0.5     # Edge length constraint weight
W_PROC = 1.0      # Rigid Alignment weight
W_ICP  = 0.5      # Chamfer/Nearest Neighbor weight
W_TPS  = 1.0      # Non-rigid warping weight

# Paths
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
TEST_MODEL_PATH = CHECKPOINT_PATH

LANDMARKS_CSV = "./data/Femur_Landmarks.csv"
EDGES_CSV = "./data/Edges.csv"