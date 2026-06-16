import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

import config
from models import LandmarkCompletionModel

# --------------------------------------------------
# Build single-sample graph (NO config switching here)
# --------------------------------------------------
def build_single_sample(
    known_landmarks,   # dict {node_id: [x,y,z]}
    side               # 0 = left, 1 = right
):
    N = config.NUM_NODES

    # -----------------------------
    # Node positions + known mask
    # -----------------------------
    pos = torch.zeros((N, 3), dtype=torch.float32)
    known_mask = torch.zeros(N, dtype=torch.bool)

    for idx, coord in known_landmarks.items():
        pos[idx] = torch.tensor(coord, dtype=torch.float32)
        known_mask[idx] = True

    # -----------------------------
    # Chirality normalization
    # -----------------------------
    if side == 0:  # LEFT
        pos[:, 0] *= -1

    # -----------------------------
    # Edge index (from CSV)
    # -----------------------------
    edges_df = pd.read_csv(config.EDGES_CSV)
    v1 = edges_df["V1"].values - 1
    v2 = edges_df["V2"].values - 1

    edge_index = torch.tensor(
        np.vstack([v1, v2]),
        dtype=torch.long
    )

    # -----------------------------
    # Edge attributes (Euclidean)
    # -----------------------------
    edge_attr = torch.norm(
        pos[v1] - pos[v2], dim=1
    ).unsqueeze(1)

    # -----------------------------
    # PyG batching semantics (B = 1)
    # -----------------------------
    batch_idx = torch.zeros(N, dtype=torch.long)
    side_tensor = torch.tensor([side], dtype=torch.long)

    data = Data(
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        known_mask=known_mask,
        batch=batch_idx,
        side=side_tensor,
        original_side=side_tensor.clone()
    )

    return data

# --------------------------------------------------
# Single-sample inference (evaluator-style)
# --------------------------------------------------
def infer_single_sample(
    known_landmarks,
    bone_type,
    side
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.set_bone_config(bone_type)

    N = config.NUM_NODES
    valid_ids = set(config.KNOWN_IDS)

    input_ids = set(known_landmarks.keys())
    if not input_ids.issubset(valid_ids):
        raise ValueError(
            f"Invalid landmark IDs for {bone_type.upper()}.\n"
            f"Expected subset of {sorted(valid_ids)}, got {sorted(input_ids)}"
        )

    model = LandmarkCompletionModel().to(device)
    model.load_state_dict(
        torch.load(config.CHECKPOINT_PATH, map_location=device)
    )
    model.eval()

    data = build_single_sample(
        known_landmarks=known_landmarks,
        side=side
    ).to(device)

    with torch.no_grad():
        pred_mm, _, confidence = model(data)

    # ALIGN TO GT SPACE (same as evaluator)
    pred_aligned = model.align_template(
        1,
        pos_target=pred_mm.view(1, N, 3),
        known_mask=data.known_mask.view(1, N)
    )

    pred = pred_aligned[0].cpu().numpy()

        # --------------------------------------------------
    # SAVE OUTPUT TO CSV
    # --------------------------------------------------
    output_filename = f"{bone_type}_{'L' if side == 0 else 'R'}_predicted.csv"

    df = pd.DataFrame({
        "node_id": np.arange(config.NUM_NODES),
        "x": pred[:, 0],
        "y": pred[:, 1],
        "z": pred[:, 2]
    })

    df.to_csv(output_filename, index=False)

    print(f"\nSaved prediction to: {output_filename}")

    if side == 0:
        pred[:, 0] *= -1

    for idx, coord in known_landmarks.items():
        pred[idx] = np.array(coord, dtype=np.float32)

    return pred

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    # YOUR PROVIDED TIBIA LANDMARKS
    # known_landmarks = {
    #     3: [-336.682959011335,	279.446727005061,	-919.401637290738],
    #     4: [-375.28029327424,	292.732612007749,	-916.311659415108],
    #     9: [-348.071791724773,	304.422899573105,	-1250.77143215399],
    #     10: [-370.784625902033,	263.278782465427,	-921.186224864822]
    # }

    known_landmarks = {
        0: [-71.6971095496993,	620.827162361965,	-798.2977916151],
        1: [-61.6564090362143,	404.322884567993,	-475.472945134407],
        2: [-27.4008447713944,	427.645873566119,	-481.793823659386],
        3: [-93.3676867599639,	397.137408158682,	-501.143295846327]
    }

#     -71.6971095496993	620.827162361965	-798.2977916151
# -61.6564090362143	404.322884567993	-475.472945134407
# -27.4008447713944	427.645873566119	-481.793823659386
# -93.3676867599639	397.137408158682	-501.143295846327


    # bone_type = "tibia"
    # side = 1  # 0 = left, 1 = right

    bone_type =  "Femur"
    side = 1

    pred = infer_single_sample(
        known_landmarks=known_landmarks,
        bone_type=bone_type,
        side=side
    )

    print("\n=== TIBIA SINGLE-SAMPLE PREDICTION ===")
    for i in range(config.NUM_NODES):
        tag = "*" if i not in config.KNOWN_IDS else " "
        print(f"Node {i:02d}{tag}: {pred[i]}")
