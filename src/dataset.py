import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor

class RoadSegDataset(Dataset):
    """
    PyTorch dataset for road segmentation.
    Loads full-frame image + mask pairs, preprocesses them for SegFormer.
    """

    def __init__(self, img_paths, mask_paths,
                 model_name='nvidia/segformer-b0-finetuned-cityscapes-512-1024'):
        """
        Args:
            img_paths (List[str]): Paths to input images (JPEG/PNG).
            mask_paths (List[str]): Paths to binary mask PNGs (0=bg,255=road).
            model_name (str): HuggingFace checkpoint to load the image processor.
        """
        assert len(img_paths) == len(mask_paths), "Images and masks must match"
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.processor = SegformerImageProcessor.from_pretrained(model_name)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image and convert BGR→RGB
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask and normalize to 0/1
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) // 255

        # Use the SegFormer processor to resize and normalize the image
        enc = self.processor(img, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)    # [C, H, W]

        # Resize mask to match model input
        _, _, H, W = enc["pixel_values"].shape
        mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # Convert mask to tensor
        labels = torch.from_numpy(mask_resized).long()  # [H, W]

        return {
            "pixel_values": pixel_values,  # float tensor [C, H, W]
            "labels": labels               # long tensor [H, W]
        }
