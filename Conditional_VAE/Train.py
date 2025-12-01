# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# import os
# import random
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from Config import ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, EPOCHS, BATCH_SIZE, LR, KL_WEIGHT_BASE, VAL_SPLIT, NUM_WORKERS, CHECKPOINT_DIR, VIS_DIR, DEVICE
# from Dataloader import FemurVAEDataset, collate_fn
# from Model import FemurVAE
# from Utilities import vae_loss_raw

# def train():
#     full_ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV)
#     n_samples = len(full_ds)
#     indices = list(range(n_samples))
#     random.shuffle(indices)
#     n_val = max(1, int(n_samples * VAL_SPLIT))
#     train_indices = indices[:-n_val]
#     val_indices = indices[-n_val:]

#     train_ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, indices=train_indices)
#     val_ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, indices=val_indices)

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

#     # Model and optimizer
#     model = FemurVAE().to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     train_losses = []
#     val_losses = []
#     epochs_list = []
#     best_val = float('inf')

#     for epoch in range(1, EPOCHS + 1):
#         kl_w = min(epoch / 50.0, 1.0)

#         # Train
#         model.train()
#         train_loss_total = 0.0
#         n_train_batches = 0
#         pbar_train = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}")
#         for batch in pbar_train:
#             pts = batch['points'].to(DEVICE)
#             cond = batch['cond'].to(DEVICE)
#             recon, mu, logvar = model(pts, cond)
#             loss, recon_l, kl_l = vae_loss_raw(recon, pts, mu, logvar, KL_WEIGHT_BASE * kl_w)
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             optimizer.step()
#             train_loss_total += loss.item()
#             n_train_batches += 1
#             pbar_train.set_postfix(loss=loss.item(), recon=f"{recon_l:.4f}", kl=f"{kl_l:.4f}")

#         avg_train = train_loss_total / max(1, n_train_batches)
#         train_losses.append(avg_train)

#         # Val
#         model.eval()
#         val_loss_total = 0.0
#         n_val_batches = 0
#         pbar_val = tqdm(val_loader, desc=f"[Val] Epoch {epoch}/{EPOCHS}")
#         with torch.no_grad():
#             for batch in pbar_val:
#                 pts = batch['points'].to(DEVICE)
#                 cond = batch['cond'].to(DEVICE)
#                 recon, mu, logvar = model(pts, cond)
#                 loss, recon_l, kl_l = vae_loss_raw(recon, pts, mu, logvar, KL_WEIGHT_BASE * kl_w)
#                 val_loss_total += loss.item()
#                 n_val_batches += 1
#                 pbar_val.set_postfix(loss=loss.item(), recon=f"{recon_l:.4f}")

#         avg_val = val_loss_total / max(1, n_val_batches)
#         val_losses.append(avg_val)
#         epochs_list.append(epoch)

#         print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | KL_w={kl_w:.3f}")

#         if avg_val < best_val:
#             best_val = avg_val
#             torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vae_best.pt"))

#         if epoch % 10 == 0:
#             torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"vae_e{epoch}.pt"))

#     return epochs_list, train_losses, val_losses

# if __name__ == '__main__':
#     print("Starting training...")
#     epochs, train_l, val_l = train()
#     print("Training complete. Plotting...")

#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, train_l, 'o-', label='Train Loss', linewidth=2)
#     plt.plot(epochs, val_l, 's-', label='Val Loss', linewidth=2)
#     plt.title('Femur CVAE Loss Convergence')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{VIS_DIR}/loss_curve.png", dpi=200)
#     plt.show()

#     train_drop = (train_l[0] - train_l[-1]) / train_l[0] * 100 if len(train_l) > 1 and train_l[0] != 0 else 0.0
#     val_drop = (val_l[0] - val_l[-1]) / val_l[0] * 100 if len(val_l) > 1 and val_l[0] != 0 else 0.0
#     gap = abs(train_l[-1] - val_l[-1])
#     print(f"Summary: Train drop {train_drop:.1f}% | Val drop {val_drop:.1f}% | Final gap {gap:.4f}")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from Config import ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, EPOCHS, BATCH_SIZE, LR, KL_WEIGHT_BASE, VAL_SPLIT, NUM_WORKERS, CHECKPOINT_DIR, VIS_DIR, DEVICE
from Dataloader import FemurVAEDataset, collate_fn
from Model import FemurVAE

from torch.optim.lr_scheduler import CosineAnnealingLR

# def vae_loss_raw(recon, pts, mu, logvar, kl_weight, free_bits=10.0):  # Higher for diversity
#     recon_l = F.smooth_l1_loss(recon, pts, reduction='mean')
#     kl_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     kl_l = torch.sum(torch.max(kl_l / kl_l.size(0), torch.tensor(free_bits * np.log(2), device=kl_l.device)))
#     loss = recon_l + kl_weight * kl_l
#     return loss, recon_l, kl_l

def vae_loss_raw(recon, pts, mu, logvar, kl_weight, free_bits=10.0):
    recon_l = F.smooth_l1_loss(recon, pts, reduction='mean')
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
    kl_capped = torch.max(kl_per_dim, torch.tensor(free_bits * np.log(2), device=kl_per_dim.device))
    kl_l = torch.sum(kl_capped)
    loss = recon_l + kl_weight * kl_l
    return loss, recon_l, kl_l

def train():
    full_ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV)
    n_samples = len(full_ds)
    indices = list(range(n_samples))
    random.shuffle(indices)
    n_val = max(1, int(n_samples * VAL_SPLIT))
    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]

    train_ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, indices=train_indices)
    val_ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV, indices=val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # FIXED: cond_dim=42
    model = FemurVAE(n_points=1024, latent_dim=256, cond_dim=42).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Compute real cond stats
    print("Computing cond stats...")
    all_cond = torch.stack([torch.tensor(val_ds[i]['cond']) for i in range(len(val_ds))])
    cond_mean = all_cond.mean(dim=0)
    cond_std = all_cond.std(dim=0) + 1e-6
    print(f"Cond mean: {cond_mean[:3]}..., std: {cond_std[:3]}...")
    
    train_losses, val_losses, epochs_list = [], [], []
    best_val = float('inf')
    patience, early_stop = 20, False

    for epoch in range(1, EPOCHS + 1):
        # Cyclical KL for diversity
        kl_w = KL_WEIGHT_BASE * ((0.5 * np.sin(2 * np.pi * epoch / 10) + 1.5) * min(epoch / 50.0, 1.0))

        # Train
        model.train()
        train_loss_total, n_train_batches = 0.0, 0
        active_latents_avg = 0.0
        pbar_train = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}")
        for batch in pbar_train:
            pts = batch['points'].to(DEVICE)
            cond = batch['cond'].to(DEVICE)
            
            # Norm pts (per-sample), cond (global)
            pts_mean = pts.mean(dim=1, keepdim=True)
            pts_std = pts.std(dim=1, keepdim=True) + 1e-6
            pts_norm = (pts - pts_mean) / pts_std
            cond_norm = (cond - cond_mean.to(DEVICE)) / cond_std.to(DEVICE)
            
            recon, mu, logvar = model(pts_norm, cond_norm)
            loss, recon_l, kl_l = vae_loss_raw(recon, pts_norm, mu, logvar, kl_w)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Active latents
            active = (torch.abs(kl_l / kl_l.numel()) > 0.01).float().mean().item()
            active_latents_avg += active
            
            train_loss_total += loss.item()
            n_train_batches += 1
            pbar_train.set_postfix(kl=f"{kl_l:.1f}", loss=f"{loss:.1f}", recon=f"{recon_l:.1f}", active=f"{active:.1%}")

        avg_train = train_loss_total / max(1, n_train_batches)
        avg_active = active_latents_avg / n_train_batches
        train_losses.append(avg_train)

        # Val
        model.eval()
        val_loss_total, n_val_batches = 0.0, 0
        pbar_val = tqdm(val_loader, desc=f"[Val] Epoch {epoch}/{EPOCHS}")
        with torch.no_grad():
            for batch in pbar_val:
                pts = batch['points'].to(DEVICE)
                cond = batch['cond'].to(DEVICE)
                
                pts_mean = pts.mean(dim=1, keepdim=True)
                pts_std = pts.std(dim=1, keepdim=True) + 1e-6
                pts_norm = (pts - pts_mean) / pts_std
                cond_norm = (cond - cond_mean.to(DEVICE)) / cond_std.to(DEVICE)
                
                recon, mu, logvar = model(pts_norm, cond_norm)
                loss, recon_l, kl_l = vae_loss_raw(recon, pts_norm, mu, logvar, kl_w)
                
                val_loss_total += loss.item()
                n_val_batches += 1
                pbar_val.set_postfix(loss=f"{loss:.1f}", recon=f"{recon_l:.1f}")

        avg_val = val_loss_total / max(1, n_val_batches)
        val_losses.append(avg_val)
        epochs_list.append(epoch)
        scheduler.step()

        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train:.3f} | Val Loss: {avg_val:.3f} | KL_w={kl_w:.3f} | Active Latents: {avg_active:.1%}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vae_best.pt"))
            patience = 20
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                early_stop = True
                break

        # Prior sample viz (fixed indentation, denorm with avg stats)
        if epoch % 5 == 0:
            with torch.no_grad():
                z = torch.randn(1, model.latent_dim).to(DEVICE)
                rand_cond = (torch.rand(1, model.cond_dim).to(DEVICE) - 0.5) * 2
                rand_cond_norm = (rand_cond - cond_mean.to(DEVICE)) / cond_std.to(DEVICE)
                sample_norm = model.decode(z, rand_cond_norm)
                # Avg denorm stats (from ds)
                avg_pts_mean = torch.zeros(1, 1, 3).to(DEVICE)  # Assume centered
                avg_pts_std = torch.ones(1, 1, 3).to(DEVICE) * 100.0  # ~100mm scale
                sample = sample_norm * avg_pts_std + avg_pts_mean
                sample_np = sample.cpu().numpy().squeeze()
                fig = plt.figure(figsize=(8,6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(sample_np[:,0], sample_np[:,1], sample_np[:,2], s=1, c='b')
                ax.set_title(f'Prior Sample Epoch {epoch} (denormed)')
                plt.savefig(os.path.join(VIS_DIR, f"prior_sample_e{epoch}.png"), dpi=150)
                plt.close()
            print(f"Saved prior sample viz for epoch {epoch}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"vae_e{epoch}.pt"))

    if not early_stop:
        print("Completed all epochs.")

    return epochs_list, train_losses, val_losses

if __name__ == '__main__':
    print("Starting improved training...")
    epochs, train_l, val_l = train()
    print("Training complete. Plotting...")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_l, 'o-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_l, 's-', label='Val Loss', linewidth=2)
    plt.title('Improved Femur CVAE Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{VIS_DIR}/improved_loss_curve.png", dpi=200)
    plt.show()

    train_drop = (train_l[0] - train_l[-1]) / train_l[0] * 100 if len(train_l) > 1 and train_l[0] != 0 else 0.0
    val_drop = (val_l[0] - val_l[-1]) / val_l[0] * 100 if len(val_l) > 1 and val_l[0] != 0 else 0.0
    gap = abs(train_l[-1] - val_l[-1])
    print(f"Summary: Train drop {train_drop:.1f}% | Val drop {val_drop:.1f}% | Final gap {gap:.4f}")