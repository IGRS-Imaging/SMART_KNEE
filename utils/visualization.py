# utils/visualization.py

import torch
import numpy as np
import plotly.graph_objects as go

def _to_np(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        return tensor_or_array.detach().cpu().numpy()
    return np.array(tensor_or_array)

def _generate_edge_lists(pos, edge_index):
    x_lines, y_lines, z_lines = [], [], []
    if edge_index.shape[0] == 2 and edge_index.shape[1] > 2:
         edge_iter = edge_index.T
    else:
         edge_iter = edge_index

    for u, v in edge_iter:
        u, v = int(u), int(v)
        x_lines.extend([pos[u, 0], pos[v, 0], None])
        y_lines.extend([pos[u, 1], pos[v, 1], None])
        z_lines.extend([pos[u, 2], pos[v, 2], None])
    return x_lines, y_lines, z_lines

def _generate_hover_text(pos, label):
    hover_text = []
    for i in range(len(pos)):
        x, y, z = pos[i]
        txt = f"<b>Node: {i}</b><br>{label}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}"
        hover_text.append(txt)
    return hover_text

def visualize_shapes(pos_gt, pos_init, pos_pred, edge_index, title="Shape Visualization"):
    # 1. Data Prep
    pos_gt_np = _to_np(pos_gt)
    pos_pred_np = _to_np(pos_pred)
    pos_init_np = _to_np(pos_init) if pos_init is not None else None
    edge_index_np = _to_np(edge_index)
    
    num_nodes = pos_gt_np.shape[0]
    node_labels = [str(i) for i in range(num_nodes)] # ["0", "1", "2", ...]
    
    fig = go.Figure()

    # =========================================
    # 1. Ground Truth (Green)
    # =========================================
    fig.add_trace(go.Scatter3d(
        x=pos_gt_np[:, 0], y=pos_gt_np[:, 1], z=pos_gt_np[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='green', opacity=0.8),
        text=node_labels, 
        textposition="top center", # GT labels on TOP
        textfont=dict(size=10, color='green'),
        name='GT Nodes',
        hoverinfo="text",
        hovertext=_generate_hover_text(pos_gt_np, "Ground Truth")
    ))

    # GT Edges
    gt_x, gt_y, gt_z = _generate_edge_lists(pos_gt_np, edge_index_np)
    fig.add_trace(go.Scatter3d(
        x=gt_x, y=gt_y, z=gt_z,
        mode='lines',
        line=dict(color='green', width=2, dash='dot'),
        opacity=0.5,
        name='GT Edges',
        hoverinfo='none'
    ))

    # =========================================
    # 2. Prediction (Blue)
    # =========================================
    # Pred Nodes - NOW WITH TEXT
    fig.add_trace(go.Scatter3d(
        x=pos_pred_np[:, 0], y=pos_pred_np[:, 1], z=pos_pred_np[:, 2],
        mode='markers+text', # Changed from 'markers' to 'markers+text'
        marker=dict(size=7, color='blue', symbol='diamond', opacity=1.0),
        text=node_labels,
        textposition="bottom center", # Pred labels on BOTTOM (to avoid overlap)
        textfont=dict(size=10, color='blue'),
        name='Pred Nodes',
        hoverinfo="text",
        hovertext=_generate_hover_text(pos_pred_np, "Prediction")
    ))

    # Pred Edges
    pred_x, pred_y, pred_z = _generate_edge_lists(pos_pred_np, edge_index_np)
    fig.add_trace(go.Scatter3d(
        x=pred_x, y=pred_y, z=pred_z,
        mode='lines',
        line=dict(color='blue', width=3),
        opacity=0.8,
        name='Pred Edges',
        hoverinfo='none'
    ))
    
    # =========================================
    # 3. Initialization (Orange)
    # =========================================
    if pos_init_np is not None:
        fig.add_trace(go.Scatter3d(
            x=pos_init_np[:, 0], y=pos_init_np[:, 1], z=pos_init_np[:, 2],
            mode='markers',
            marker=dict(size=5, color='orange', opacity=0.6),
            name='Init Nodes',
            hoverinfo="text",
            hovertext=_generate_hover_text(pos_init_np, "Initialization"),
            visible='legendonly' 
        ))

    # =========================================
    # Layout
    # =========================================
    fig.update_layout(
        title=title,
        width=1200,
        height=800,
        scene=dict(
            aspectmode='data', 
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            bgcolor='white'
        ),
        legend=dict(x=0.7, y=0.9, bgcolor='rgba(255,255,255,0.8)')
    )

    fig.show()