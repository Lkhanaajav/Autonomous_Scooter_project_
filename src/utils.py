#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def combine_masks(pseudo_dir, hand_dir, out_dir):
    """
    Merge teacher pseudo-labels with hand annotations.
    Hand masks override pseudo masks wherever they exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in tqdm(os.listdir(pseudo_dir), desc="Combining masks"):
        if not name.lower().endswith('.png'):
            continue

        pseudo_path = os.path.join(pseudo_dir, name)
        hand_path   = os.path.join(hand_dir,   name)

        # Load pseudo mask (boolean)
        m1 = cv2.imread(pseudo_path, cv2.IMREAD_GRAYSCALE)
        if m1 is None:
            print(f"⚠️  Could not load pseudo mask {pseudo_path}, skipping")
            continue
        m1 = m1 > 0

        # Load hand mask if available, otherwise zero everywhere
        if os.path.exists(hand_path):
            m2 = cv2.imread(hand_path, cv2.IMREAD_GRAYSCALE)
            if m2 is None:
                print(f"⚠️  Could not load hand mask {hand_path}, treating as empty")
                m2 = np.zeros_like(m1, dtype=bool)
            else:
                m2 = m2 > 0
        else:
            m2 = np.zeros_like(m1, dtype=bool)

        # Combine: hand labels take precedence
        merged = np.where(m2, True, m1).astype(np.uint8) * 255

        out_path = os.path.join(out_dir, name)
        cv2.imwrite(out_path, merged)
    print(f"✅ Finished combining masks → {out_dir}")

def overlay_images(img_dir, mask_dir, out_dir):
    """
    Create visual check images by overlaying masks onto frames.
    Road pixels in mask are painted green on the original image.
    """
    os.makedirs(out_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(img_dir), desc="Overlaying images"):
        if not (img_name.lower().endswith('.jpg') or img_name.lower().endswith('.png')):
            continue

        img_path  = os.path.join(img_dir, img_name)
        img       = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Could not load image {img_path}, skipping")
            continue

        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"⚠️  No mask found for {img_name}, skipping overlay")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️  Could not load mask {mask_path}, skipping overlay")
            continue

        overlay = img.copy()
        overlay[mask > 0] = (0, 255, 0)  # green on road
        vis = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, vis)
    print(f"✅ Finished overlays → {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility: combine hand & pseudo masks, or overlay masks on images."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("combine_masks", help="Merge pseudo and hand masks")
    p1.add_argument("--pseudo", required=True, help="Directory of pseudo masks (.png)")
    p1.add_argument("--hand",   required=True, help="Directory of hand masks (.png)")
    p1.add_argument("--out",    required=True, help="Output directory for merged masks")

    p2 = sub.add_parser("overlay_images", help="Overlay masks onto images for QA")
    p2.add_argument("--img_dir",  required=True, help="Directory of original images")
    p2.add_argument("--mask_dir", required=True, help="Directory of masks (.png)")
    p2.add_argument("--out",      required=True, help="Output directory for overlays")

    args = parser.parse_args()

    if args.cmd == "combine_masks":
        combine_masks(args.pseudo, args.hand, args.out)
    elif args.cmd == "overlay_images":
        overlay_images(args.img_dir, args.mask_dir, args.out)
