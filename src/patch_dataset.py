import os, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
# def make_patches(img_dir, mask_dir, out_dir, size=64, stride=32):
#     os.makedirs(out_dir, exist_ok=True)
#     idx = 0
#     for img_name in os.listdir(img_dir):
#         print(f"→ Processing {img_name}")
#         img = cv2.imread(os.path.join(img_dir, img_name))
#         mask_name = img_name.rsplit('.',1)[0] + '.png'
#         mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
#         h, w = mask.shape
#         y_steps = (h - size) // stride + 1
#         x_steps = (w - size) // stride + 1
#         print(f"   will generate {y_steps * x_steps} patches ({y_steps}×{x_steps})")
#         for y in range(0, h-size+1, stride):
#             for x in range(0, w-size+1, stride):
#                 patch = img[y:y+size, x:x+size]
#                 mpatch = mask[y:y+size, x:x+size]
#                 label = 1 if (mpatch>0).sum()/mpatch.size > 0.5 else 0
#                 np.savez(os.path.join(out_dir, f"patch_{idx}.npz"), img=patch, lbl=label)
#                 idx += 1
#     print(f"✅ Finished making {idx} patches into {out_dir}")

def make_patches(img_dir, mask_dir, out_dir,
                         size=64, frame_step=100, patches_per_frame=5):
    os.makedirs(out_dir, exist_ok=True)
    idx = 0

    # 1) pick only every frame_step frame
    frames = sorted(os.listdir(img_dir))
    sampled = frames[::frame_step]

    for img_name in sampled:
        img_path  = os.path.join(img_dir, img_name)
        mask_name = img_name.rsplit('.',1)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)

        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        h, w = mask.shape
        for _ in range(patches_per_frame):
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            patch  = img[y:y+size, x:x+size]
            mpatch = mask[y:y+size, x:x+size]
            label  = 1 if (mpatch>0).sum()/mpatch.size > 0.5 else 0
            np.savez(os.path.join(out_dir, f"patch_{idx}.npz"),
                     img=patch, lbl=label)
            idx += 1

    print(f"✅ Generated {idx} patches from {len(sampled)} frames")
class PatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.files = [os.path.join(patch_dir,f) for f in os.listdir(patch_dir)]
        self.transform = transform or ToTensor()

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img, lbl = data['img'], int(data['lbl'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {'pixel_values': self.transform(img), 'labels': torch.tensor(lbl)}