from Dataloader import FemurVAEDataset
from Config import ROOT_DIR, LANDMARKS_CSV, EDGES_CSV

ds = FemurVAEDataset(ROOT_DIR, LANDMARKS_CSV, EDGES_CSV)
sample = ds[0]
print("Points shape:", sample['points'].shape)  # Expect (1024, 3)
print("Cond shape:", sample['cond'].shape)      # Expect (42,) or (1,42)
print("Cond dim:", sample['cond'].shape[-1] if len(sample['cond'].shape) > 0 else 1)