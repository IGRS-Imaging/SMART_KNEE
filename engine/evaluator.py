#engine/evaluator.py
import torch
import numpy as np
from tqdm import tqdm
import config
from utils.visualization import visualize_shapes

def evaluate_model(model, loader, device, return_detailed=False, verbose=True):
    """
    Evaluates the model on the given loader.
    
    Args:
        return_detailed (bool): If True, returns (mean, std, per_node_stats).
        verbose (bool): If True, prints the detailed ASCII table and progress bar. 
                        Set False for cleaner training loops.
    """
    model.eval()
    
    all_errors = []
    node_errors_tracker = {i: [] for i in range(config.NUM_NODES)} 
    
    # Logic for Random Visualization (Only if verbose and detailed)
    VISUALIZE_LIMIT = 5
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
    
    # Disable tqdm if not verbose to keep logs clean
    iterator = tqdm(loader, desc="Evaluating", ncols=80) if verbose else loader
    
    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device)
            
            # Forward pass
            pred_mm, _ = model(batch)
            
            # Reshape
            B = batch.num_graphs
            N = config.NUM_NODES
            pred = pred_mm.view(B, N, 3)
            target = batch.pos.view(B, N, 3)
            
            known_mask = batch.known_mask.view(B, N)
            unknown_mask = ~known_mask
            
            # (B, N) Euclidean Error
            diff = (pred - target).norm(dim=-1)
            
            # Store errors
            for b in range(B):
                # Shape error (unknown nodes only)
                mask_b = unknown_mask[b]
                if mask_b.sum() > 0:
                    shape_err = diff[b][mask_b].mean().item()
                    all_errors.append(shape_err)
                
                # Per-node error tracking
                for n in range(N):
                    err_n = diff[b, n].item()
                    node_errors_tracker[n].append(err_n)

                # --- VISUALIZATION TRIGGER ---
                if return_detailed and verbose and (current_sample_idx in indices_to_visualize):
                    print(f"Visualizing Sample {current_sample_idx} (Error: {shape_err:.2f}mm)")
                    pos_gt_np = target[b].cpu().numpy()
                    pos_pred_np = pred[b].cpu().numpy()
                    edge_idx_np = batch.edge_index.cpu().numpy()
                    current_node_errors = diff[b].cpu().numpy()
                    
                    visualize_shapes(
                        pos_gt=pos_gt_np,
                        pos_pred=pos_pred_np,
                        edge_index=edge_idx_np,
                        node_errors=current_node_errors,
                        title=f"Sample {current_sample_idx} | Mean Err: {shape_err:.2f}mm"
                    )
                
                current_sample_idx += 1

    # --- REPORTING ---
    all_errors = np.array(all_errors)
    mean_err = np.mean(all_errors) if len(all_errors) > 0 else 0
    std_err = np.std(all_errors) if len(all_errors) > 0 else 0
    
    per_node_stats = {}
    for n in range(config.NUM_NODES):
        n_errs = np.array(node_errors_tracker[n])
        n_mean = np.mean(n_errs) if len(n_errs) > 0 else 0
        n_std = np.std(n_errs) if len(n_errs) > 0 else 0
        per_node_stats[n] = {"mean": n_mean, "std": n_std}

    # Only print the big table if verbose is True
    if verbose:
        print(f"\n===== EVALUATION REPORT =====")
        print(f"Overall Mean Error (Unknowns): {mean_err:.4f} mm")
        print(f"Standard Deviation:            {std_err:.4f} mm")
        print(f"-----------------------------")
        print(f"| Node ID | Mean Err (mm) | Std Dev (mm) |")
        print(f"|---------|---------------|--------------|")
        
        for n in range(config.NUM_NODES):
            marker = "*" if n not in config.KNOWN_IDS else "" 
            stats = per_node_stats[n]
            print(f"| {str(n)+marker:7} | {stats['mean']:13.4f} | {stats['std']:12.4f} |")
        
        print(f"-----------------------------")
        print(f"* = Target Unknown Node\n")

    if return_detailed:
        return mean_err, std_err, per_node_stats
    return mean_err