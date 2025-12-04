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

def _generate_hover_text(pos, label, errors=None):
    hover_text = []
    for i in range(len(pos)):
        x, y, z = pos[i]
        err_str = f"<br><b>Error: {errors[i]:.3f} mm</b>" if errors is not None else ""
        txt = f"<b>Node: {i}</b><br>{label}{err_str}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}"
        hover_text.append(txt)
    return hover_text

def visualize_shapes(pos_gt, pos_pred, edge_index, node_errors=None, title="Shape Visualization", subject_id=None):
    """
    Visualizes GT and Prediction.
    node_errors: (N,) numpy array of errors in mm. Used for coloring pred nodes.
    subject_id: (str) Optional subject identifier to display in title.
    """
    # 1. Data Prep
    pos_gt_np = _to_np(pos_gt)
    pos_pred_np = _to_np(pos_pred)
    edge_index_np = _to_np(edge_index)
    node_errors_np = _to_np(node_errors) if node_errors is not None else None
    
    num_nodes = pos_gt_np.shape[0]
    node_labels = [str(i) for i in range(num_nodes)] 
    
    fig = go.Figure()

    # =========================================
    # 1. Ground Truth (Grey Ghost / Wireframe)
    # =========================================
    # GT Edges
    gt_x, gt_y, gt_z = _generate_edge_lists(pos_gt_np, edge_index_np)
    fig.add_trace(go.Scatter3d(
        x=gt_x, y=gt_y, z=gt_z,
        mode='lines',
        line=dict(color='grey', width=2, dash='dot'),
        opacity=0.5,
        name='GT Structure',
        hoverinfo='none'
    ))

    # GT Nodes (Small markers)
    fig.add_trace(go.Scatter3d(
        x=pos_gt_np[:, 0], y=pos_gt_np[:, 1], z=pos_gt_np[:, 2],
        mode='markers',
        marker=dict(size=4, color='grey', opacity=0.5),
        name='GT Nodes',
        hoverinfo="text",
        hovertext=_generate_hover_text(pos_gt_np, "Ground Truth")
    ))

    # =========================================
    # 2. Prediction (Color-Coded by Error)
    # =========================================
    
    # Determine colors
    if node_errors_np is not None:
        # Scale: 0mm = Green, 5mm = Yellow, 10mm+ = Red
        cmin, cmax = 0, 15 
        marker_color = node_errors_np
        colorscale = 'RdYlGn_r' # Reverse Red-Yellow-Green so Green is low error
        colorbar_title = 'Error (mm)'
    else:
        marker_color = 'blue'
        colorscale = None
        cmin, cmax = None, None
        colorbar_title = None

    # Pred Nodes
    fig.add_trace(go.Scatter3d(
        x=pos_pred_np[:, 0], y=pos_pred_np[:, 1], z=pos_pred_np[:, 2],
        mode='markers+text', 
        marker=dict(
            size=8, 
            color=marker_color, 
            colorscale=colorscale, 
            cmin=cmin, cmax=cmax,
            showscale=True,
            colorbar=dict(title=colorbar_title, x=0.85),
            symbol='diamond', 
            opacity=1.0
        ),
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=10, color='black'),
        name='Prediction',
        hoverinfo="text",
        hovertext=_generate_hover_text(pos_pred_np, "Prediction", node_errors_np)
    ))

    # Pred Edges
    pred_x, pred_y, pred_z = _generate_edge_lists(pos_pred_np, edge_index_np)
    fig.add_trace(go.Scatter3d(
        x=pred_x, y=pred_y, z=pred_z,
        mode='lines',
        line=dict(color='black', width=3),
        opacity=0.8,
        name='Pred Structure',
        hoverinfo='none'
    ))
    
    # =========================================
    # Layout
    # =========================================
    
    # Construct Title
    plot_title = title
    if subject_id is not None:
        plot_title = f"{title} | Subject: {subject_id}"

    fig.update_layout(
        title=plot_title,
        width=1000,
        height=700,
        scene=dict(
            aspectmode='data', 
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            bgcolor='white'
        ),
        legend=dict(x=0.05, y=0.9)
    )

    fig.show()