import cv2
import numpy as np

"""
BEV Transform Utility (Segmentation Mask to BEV)
- Loads 4 source points (from select_bev_points.py, e.g., bev_src_points.npy)
- Defines 4 destination points for BEV space
- Computes homography
- Warps a segmentation mask to BEV

Instructions:
1. Use select_bev_points.py to save your 4 source points as bev_src_points.npy
2. Set BEV_SIZE as desired (e.g., 400x400)
3. Place your segmentation mask as test_mask.png (same size as the frame used for point selection)
4. Run this script to see BEV transformation on the mask
"""

# --- CONFIG ---
SRC_POINTS_PATH = 'bev_src_points.npy'  # 4 points from select_bev_points.py
BEV_SIZE = (400, 400)  # (width, height) of BEV output
MASK_PATH = 'test_mask.png'  # Path to the segmentation mask image

# --- Load source points ---
src_points = np.load(SRC_POINTS_PATH)  # shape (4, 2)

# --- Define destination points (rectangle in BEV space) ---
dst_points = np.array([
    [0, BEV_SIZE[1]],         # bottom-left
    [BEV_SIZE[0], BEV_SIZE[1]],  # bottom-right
    [BEV_SIZE[0], 0],         # top-right
    [0, 0],                   # top-left
], dtype=np.float32)

# --- Compute homography ---
H, _ = cv2.findHomography(src_points.astype(np.float32), dst_points)

def warp_to_bev(img, H, bev_size):
    """
    Warps an image or mask to BEV using homography H.
    img: input image or mask
    H: homography matrix
    bev_size: (width, height)
    Returns: BEV image/mask
    """
    return cv2.warpPerspective(img, H, bev_size)

if __name__ == '__main__':
    # --- DEMO: Load and warp a segmentation mask only ---
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f'Please provide {MASK_PATH} for demo, or update the path.')
        exit(1)

    bev_mask = warp_to_bev(mask, H, BEV_SIZE)

    # Show results
    cv2.imshow('Original Mask', mask)
    cv2.imshow('BEV Mask', bev_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    cv2.imwrite('bev_mask.png', bev_mask)
    print('BEV mask saved as bev_mask.png') 