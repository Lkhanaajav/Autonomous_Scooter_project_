# src/patch_utils.py

import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def make_patches(
    img_dir: str,
    mask_dir: str,
    out_dir: str,
    size: int = 64,
    frame_step: int = 100,
    patches_per_frame: int = 5
):
    """
    Randomly sample 'patches_per_frame' patches of shape (size×size) 
    from every 'frame_step'-th image in img_dir, using corresponding masks 
    to label each patch as road (1) if >50% of pixels are road, else 0.

    Saves .npz files containing:
      - 'img':  H×W×3 uint8 RGB patch
      - 'lbl':  integer 0 or 1
    into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    idx = 0

    # List and sort all frame filenames
    frames = sorted(os.listdir(img_dir))
    # Only process every `frame_step`-th frame
    sampled = frames[::frame_step]

    for img_name in sampled:
        img_path  = os.path.join(img_dir, img_name)
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)

        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        h, w = mask.shape
        for _ in range(patches_per_frame):
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            patch  = img[y : y + size, x : x + size]
            mpatch = mask[y : y + size, x : x + size]
            label  = 1 if (mpatch > 0).sum() / mpatch.size > 0.5 else 0

            out_path = os.path.join(out_dir, f"patch_{idx}.npz")
            np.savez(out_path, img=patch, lbl=label)
            idx += 1

    print(f"✅ Generated {idx} patches from {len(sampled)} frames")


class PatchDataset(Dataset):
    """
    PyTorch Dataset for loading 64×64 patches saved as .npz files.
    Each .npz must contain:
      - 'img':  shape (64,64,3), dtype uint8 (BGR)
      - 'lbl':  int 0 or 1
    Returns:
      - 'pixel_values': torch.FloatTensor [3×64×64], normalized to ImageNet stats
      - 'labels':        torch.LongTensor scalar (0 or 1)
    """
    def __init__(self, patch_dir: str, transform=None):
        self.files = sorted([
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir)
            if f.endswith(".npz")
        ])
        # By default, convert H×W×3 uint8 to FloatTensor and normalize
        self.transform = transform or ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img = data["img"]          # (64,64,3) uint8 BGR
        lbl = int(data["lbl"])     # 0 or 1

        # Convert BGR→RGB, then to [0,1] FloatTensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)  # FloatTensor [3,64,64] in [0,1]

        # Normalize with ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img  = (img - mean) / std

        return {"pixel_values": img, "labels": torch.tensor(lbl, dtype=torch.long)}
