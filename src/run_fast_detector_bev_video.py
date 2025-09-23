import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from fast_road_detector import FastRoadDetector, Config

"""
Video BEV Pipeline:
- Runs FastRoadDetector on each frame of a video
- Warps the segmentation mask to BEV space
- Visualizes and saves the BEV mask video
- Shows original frame and BEV mask side by side for comparison
- Saves a side-by-side comparison video

Instructions:
1. Make sure you have bev_src_points.npy (from select_bev_points.py) in the project root
2. Set BEV_SIZE as desired (e.g., 400x400)
3. Place your input video as test_video1.mp4 in the project root
4. Run this script from the project root: python src/run_fast_detector_bev_video.py
"""

# --- CONFIG ---
VIDEO_PATH = 'test_video_june_03_3.mp4'
OUTPUT_PATH = 'results/fast_detector_bev_output.mp4'
COMPARISON_PATH = 'results/fast_detector_bev_comparison.mp4'
SRC_POINTS_PATH = 'src/bev_src_points.npy'
BEV_SIZE = (400, 400)

# --- Load BEV homography ---
# IMPORTANT: src_points must be the 4 points you clicked in the original image (not the BEV rectangle)
src_points = np.load(SRC_POINTS_PATH)
dst_points = np.array([
    [0, BEV_SIZE[1]],
    [BEV_SIZE[0], BEV_SIZE[1]],
    [BEV_SIZE[0], 0],
    [0, 0],
], dtype=np.float32)
H, _ = cv2.findHomography(src_points.astype(np.float32), dst_points)

def warp_to_bev(img, H, bev_size):
    return cv2.warpPerspective(img, H, bev_size)

# --- Detector config ---
config = Config(
    video_path=VIDEO_PATH,
    model_dir="models/my-segformer-road_new",
    output_mp4=OUTPUT_PATH,
    road_id=1,
    conf_thresh=0.6,
    frame_step=1,
    use_gpu=True,
    enable_logging=False,
    enable_edge_cleaning=False,
    enable_simple_smoothing=False,
    smoothing_weight=0.2
)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, BEV_SIZE)
    comp_out = cv2.VideoWriter(COMPARISON_PATH, fourcc, fps, (BEV_SIZE[0]*2, BEV_SIZE[1]))
    detector = FastRoadDetector(config)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 1. Get segmentation mask
        mask, _ = detector.process_frame(frame)
        # 2. Warp mask to BEV
        bev_mask = warp_to_bev(mask, H, BEV_SIZE)
        # 3. Visualize (colorize for display)
        color_bev_mask = cv2.applyColorMap((bev_mask * 255).astype(np.uint8), cv2.COLORMAP_JET) if bev_mask.max() <= 1 else cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        # 4. Resize original frame to BEV size for side-by-side
        frame_resized = cv2.resize(frame, BEV_SIZE)
        # 5. Stack original and BEV mask side by side
        comparison = np.hstack((frame_resized, color_bev_mask))
        # 6. Show and save
        cv2.imshow('Original (left) | BEV Mask (right)', comparison)
        out.write(color_bev_mask)
        comp_out.write(comparison)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
    cap.release()
    out.release()
    comp_out.release()
    cv2.destroyAllWindows()
    print(f'Done! BEV mask video saved to {OUTPUT_PATH}')
    print(f'Done! Side-by-side comparison video saved to {COMPARISON_PATH}')

if __name__ == '__main__':
    main() 