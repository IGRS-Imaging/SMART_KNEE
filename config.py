import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data Settings

# # Femur
# NUM_NODES = 12
# KNOWN_IDS = [0, 2, 7]   

# Tibia
NUM_NODES = 11
KNOWN_IDS = [3, 5, 6]

# Model Architecture
FEAT_DIM = 128  # Back to original (192 was overkill)
HIDDEN_DIM = 256  # Back to original (384 was overkill)
GNN_LAYERS = 5  # Slightly increased from 4
EDGE_DIM = 1

# Training
BATCH_SIZE = 16
LR = 3e-5  # Reduced from 1e-4 for finer updates
NUM_EPOCHS = 200 

# Normalization & Template
GLOBAL_SCALE = 500.0 

# =================================== Femur =====================================
# MEAN_SHAPE_PATH = os.path.join(PROJECT_ROOT, "data", "mean_canonical_shape.npy") 
# LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "FEMUR_LANDMARKS.csv")
# EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "Femur_Edges.csv")
# CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
# TEST_MODEL_PATH = CHECKPOINT_PATH
# LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "training_log.csv") 

# ================================= Tibia =================================
MEAN_SHAPE_PATH = os.path.join(PROJECT_ROOT, "data", "mean_canonical_shape_tibia.npy") 
# LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "TIBIA_LANDMARKS_A.csv")
LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "batchreplace_lm.csv")
# EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "Tibia_EDGES.csv")
EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "batchreplace_edges.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_tibia.pt")
TEST_MODEL_PATH = CHECKPOINT_PATH
LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "training_log.csv") 

# Loss Weights
W_POS = 5.0  
W_EDGE = 2.5  
W_LOCAL_STRUCT = 3.0  
W_ANGLE = 0.8  