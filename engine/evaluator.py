import torch
import numpy as np
from tqdm import tqdm
import config
from utils.visualization import visualize_shapes

def evaluate_model(model, loader, device, return_detailed=False, verbose=True):
    """
    Evaluates the model with confidence tracking.
    """
    model.eval()
    
    all_errors = []
    node_errors_tracker = {i: [] for i in range(config.NUM_NODES)} 
    node_confidence_tracker = {i: [] for i in range(config.NUM_NODES)}
    
    # Visualization logic
    VISUALIZE_LIMIT = 36
    indices_to_visualize = set()
    
    if return_detailed and verbose:
        total_samples = len(loader.dataset)
        num_to_pick = min(VISUALIZE_LIMIT, total_samples)
        if num_to_pick > 0:
            indices_to_visualize = set(np.random.choice(
                total_samples, size=num_to_pick, replace=False
            ))
            print(f"Visualizing random indices: {sorted(list(indices_to_visualize))}")

    current_sample_idx = 0
    
    iterator = tqdm(loader, desc="Evaluating", ncols=80) if verbose else loader
    
    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device)
            
            # Forward pass with confidence
            pred_mm, _, confidence = model(batch)
            
            # Reshape
            B = batch.num_graphs
            N = config.NUM_NODES
            pred = pred_mm.view(B, N, 3)
            target = batch.pos.view(B, N, 3)
            confidence = confidence.view(B, N)
            
            known_mask = batch.known_mask.view(B, N)
            unknown_mask = ~known_mask
            
            # Calculate errors
            diff = (pred - target).norm(dim=-1)
            
            for b in range(B):
                mask_b = unknown_mask[b]
                if mask_b.sum() > 0:
                    shape_err = diff[b][mask_b].mean().item()
                    all_errors.append(shape_err)
                
                # Track per-node
                for n in range(N):
                    err_n = diff[b, n].item()
                    conf_n = confidence[b, n].item()
                    node_errors_tracker[n].append(err_n)
                    node_confidence_tracker[n].append(conf_n)

                # Visualization
                if return_detailed and verbose and (current_sample_idx in indices_to_visualize):
                    pos_gt_np = target[b].cpu().numpy()
                    pos_pred_np = pred[b].cpu().numpy()
                    edge_idx_np = batch.edge_index.cpu().numpy()
                    current_node_errors = diff[b].cpu().numpy()
                    
                    subj_id = None
                    if hasattr(batch, 'subject'):
                        if isinstance(batch.subject, list):
                            subj_id = batch.subject[b]
                        else:
                            subj_id = str(batch.subject)
                    
                    # Restore chirality for visualization
                    if hasattr(batch, 'original_side'):
                        if batch.original_side[b] == 0:
                            pos_gt_np[:, 0] *= -1
                            pos_pred_np[:, 0] *= -1
                    
                    print(f" Sample {current_sample_idx} ({subj_id}) | Err: {shape_err:.2f}mm | Low-Conf Nodes: {(confidence[b] < 0.7).sum().item()}")
                    visualize_shapes(
                        pos_gt=pos_gt_np,
                        pos_pred=pos_pred_np,
                        edge_index=edge_idx_np,
                        node_errors=current_node_errors,
                        title=f"Sample {current_sample_idx} | Err: {shape_err:.2f}mm",
                        subject_id=subj_id
                    )
                
                current_sample_idx += 1

    # --- REPORTING ---
    all_errors = np.array(all_errors)
    mean_err = np.mean(all_errors) if len(all_errors) > 0 else 0
    std_err = np.std(all_errors) if len(all_errors) > 0 else 0
    median_err = np.median(all_errors) if len(all_errors) > 0 else 0
    
    per_node_stats = {}
    for n in range(config.NUM_NODES):
        n_errs = np.array(node_errors_tracker[n])
        n_confs = np.array(node_confidence_tracker[n])
        n_mean = np.mean(n_errs) if len(n_errs) > 0 else 0
        n_std = np.std(n_errs) if len(n_errs) > 0 else 0
        n_conf_mean = np.mean(n_confs) if len(n_confs) > 0 else 0
        per_node_stats[n] = {
            "mean": n_mean, 
            "std": n_std,
            "confidence": n_conf_mean
        }

    if verbose:
        print(f"\n===== EVALUATION REPORT (OPTIMIZED MODEL) =====")
        print(f"Overall Mean Error (Unknowns):  {mean_err:.4f} mm")
        print(f"Median Error:                   {median_err:.4f} mm")
        print(f"Standard Deviation:             {std_err:.4f} mm")
        print(f"------------------------------------------------")
        print(f"| Node | Mean Err | Std Dev | Confidence |")
        print(f"|------|----------|---------|------------|")
        
        for n in range(config.NUM_NODES):
            marker = "*" if n not in config.KNOWN_IDS else "" 
            stats = per_node_stats[n]
            print(f"| {str(n)+marker:4} | {stats['mean']:8.4f} | {stats['std']:7.4f} | {stats['confidence']:10.4f} |")
        
        print(f"------------------------------------------------")
        print(f"* = Target Unknown Node")
        print(f"\nNodes with Mean Error > 4mm:")
        for n in range(config.NUM_NODES):
            if n not in config.KNOWN_IDS and per_node_stats[n]['mean'] > 4.0:
                print(f"  Node {n}: {per_node_stats[n]['mean']:.2f}mm (conf: {per_node_stats[n]['confidence']:.2f})")
        print()

    if return_detailed:
        return mean_err, std_err, per_node_stats
    return mean_err