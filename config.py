import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data Settings
NUM_NODES = 12
KNOWN_IDS = [0, 2, 7]   

# Model Architecture
FEAT_DIM = 128
HIDDEN_DIM = 256 
GNN_LAYERS = 8       
EDGE_DIM = 1

# Training
BATCH_SIZE = 16
LR = 2e-4            
NUM_EPOCHS = 300     

# Normalization & Template
GLOBAL_SCALE = 500.0 

MEAN_SHAPE_PATH = os.path.join(PROJECT_ROOT, "data", "mean_canonical_shape.npy") 
LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "Femur_Landmarks.csv")
EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "Edges.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
TEST_MODEL_PATH = CHECKPOINT_PATH
LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "training_log.csv") 

# Loss Weights (Shape Preservation Strategy)
W_POS = 10.0      # Drive the points to the correct spot
W_UNKNOWN = 1.0   # (Included in logic)

# Structural Weights
W_EDGE = 5.0      # Local rigidity
W_GLOBAL = 5.0    # NEW: Global rigidity (prevents distortion of unconnected nodes)
W_ANGLE = 100.0   # MASSIVE BOOST: Forces the orientation to match (prevents folding)

# Reduce structural weights slightly to allow more deformation
W_POS = 10.0      # Keep high
W_UNKNOWN = 1.0   

W_EDGE = 2.0      # Reduced from 5.0 (Allow some stretching)
W_GLOBAL = 2.0    # Reduced from 5.0
W_ANGLE = 50.0    # Reduced from 100.0 (Allow some bending)