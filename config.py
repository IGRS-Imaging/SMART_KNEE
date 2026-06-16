# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ================= GLOBAL SETTINGS (Shared) =================
FEAT_DIM = 128  
HIDDEN_DIM = 256  
GNN_LAYERS = 4  
EDGE_DIM = 1
BATCH_SIZE = 16
LR = 3e-5  
NUM_EPOCHS = 300 
GLOBAL_SCALE = 500.0 
W_POS = 5.0  
W_EDGE = 2.5  
W_LOCAL_STRUCT = 3.0  
W_ANGLE = 0.8  

# ================= BONE-SPECIFIC CONFIGS =================
FEMUR_CONFIG = {
    "NUM_NODES": 12,
    "KNOWN_IDS": [0, 1, 2, 3],
    "MEAN_SHAPE_FILENAME": "mean_canonical_shape_femur.npy",
    "LANDMARKS_FILENAME": "FEMUR_LANDMARKS.csv",
    "EDGES_FILENAME": "Femur_Edges.csv",
    "CHECKPOINT_FILENAME": "best_model_femur.pt",
    "NODE_IMPORTANCE": {11: 4.0, 10: 1.5, 7: 2.0, 4: 1.5}
}

TIBIA_CONFIG = {
    "NUM_NODES": 11,
    "KNOWN_IDS": [3, 4, 9, 10],
    "MEAN_SHAPE_FILENAME": "mean_canonical_shape_tibia.npy",
    "LANDMARKS_FILENAME": "batchreplace_lm.csv",
    "EDGES_FILENAME": "batchreplace_edges.csv",
    "CHECKPOINT_FILENAME": "best_model_tibia.pt",
    "NODE_IMPORTANCE": {} 
}

# ================= DEFAULTS (Placeholders) =================
# These are overwritten by set_bone_config, but declared here to satisfy imports
NUM_NODES = 12
KNOWN_IDS = [0, 1, 2, 3]
MEAN_SHAPE_PATH = ""
LANDMARKS_CSV = ""
EDGES_CSV = ""
CHECKPOINT_PATH = ""
TEST_MODEL_PATH = ""
LOG_PATH = ""
NODE_IMPORTANCE = {}
PRED_OUTPUT_DIR = ""

def set_bone_config(bone_type):
    """
    Updates the global variables in this module based on the bone type.
    """
    global NUM_NODES, KNOWN_IDS, MEAN_SHAPE_PATH, LANDMARKS_CSV
    global EDGES_CSV, CHECKPOINT_PATH, TEST_MODEL_PATH, LOG_PATH, NODE_IMPORTANCE, PRED_OUTPUT_DIR

    if bone_type.lower() == 'tibia':
        cfg = TIBIA_CONFIG
    else:
        cfg = FEMUR_CONFIG

    print(f"⚙️  Loading Configuration for: {bone_type.upper()}")

    # 1. Update Parameters
    NUM_NODES = cfg["NUM_NODES"]
    KNOWN_IDS = cfg["KNOWN_IDS"]
    NODE_IMPORTANCE = cfg["NODE_IMPORTANCE"]

    # 2. Define folder paths
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    # 3. Create outputs folder if it doesn't exist
    os.makedirs(outputs_dir, exist_ok=True)

    # 4. Update Paths
    MEAN_SHAPE_PATH = os.path.join(checkpoints_dir, cfg["MEAN_SHAPE_FILENAME"])  # checkpoints/
    LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", cfg["LANDMARKS_FILENAME"])
    EDGES_CSV = os.path.join(PROJECT_ROOT, "data", cfg["EDGES_FILENAME"])

    CHECKPOINT_PATH = os.path.join(checkpoints_dir, cfg["CHECKPOINT_FILENAME"])
    TEST_MODEL_PATH = CHECKPOINT_PATH
    LOG_PATH = os.path.join(outputs_dir, f"training_log_{bone_type}.csv")   # outputs/
    PRED_OUTPUT_DIR = outputs_dir                                             # outputs/

# Set default
set_bone_config('femur')