NUM_NODES = 12
KNOWN_IDS = [0, 2, 7]   # hip centre, medial epicondyle, lateral proximal condyle

# Model
FEAT_DIM = 128
HIDDEN_DIM = 128
GNN_LAYERS = 4
EDGE_DIM = 1

# Training
BATCH_SIZE = 8
LR = 1e-3
NUM_EPOCHS = 100
LAMBDA_SHAPE = 0.1   # weight for pairwise distance loss

CHECKPOINT_PATH = "./checkpoints/best_model.pt"
TEST_MODEL_PATH = CHECKPOINT_PATH

LANDMARKS_CSV = "./data/Femur_Landmarks.csv"
EDGES_CSV = "./data/Edges.csv"