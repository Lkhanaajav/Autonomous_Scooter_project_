import cv2
import numpy as np
import os

INSTRUCTIONS = """
INSTRUCTIONS:
- This tool lets you select 4 points in order: bottom-left, bottom-right, top-right, top-left.
- Click on the image to select each point. The coordinates will be printed and drawn.
- Press 'r' to reset points. Press 'q' to quit after selecting 4 points.
- You can use a video file (extracts first frame) or an image file for selection.
"""

print(INSTRUCTIONS)

# --- CONFIG ---
VIDEO_PATH = 'test_video_june_03_3.mp4'  # e.g., 'test_video1.mp4' or None
IMAGE_PATH = 'src/test_frame.jpg'  # fallback if VIDEO_PATH is None

# --- Extract frame from video or load image ---
if VIDEO_PATH is not None and os.path.exists(VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame from video: {VIDEO_PATH}")
    print(f"Loaded first frame from video: {VIDEO_PATH}")
else:
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    print(f"Loaded image: {IMAGE_PATH}")

# Resize image for display if too large
max_dim = 800  # Max width or height for display
h, w = img.shape[:2]
scale = 1.0
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    display_img = cv2.resize(img, (int(w * scale), int(h * scale)))
else:
    display_img = img.copy()

points = []

# Mouse callback function
def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        # Convert display coordinates back to original image coordinates
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        points.append((orig_x, orig_y))
        print(f"Point {len(points)}: ({orig_x}, {orig_y})")

cv2.namedWindow('Select 4 Points')
cv2.setMouseCallback('Select 4 Points', select_point)

while True:
    vis = display_img.copy()
    for idx, pt in enumerate(points):
        # Draw on display image (convert original coords to display coords)
        disp_x = int(pt[0] * scale)
        disp_y = int(pt[1] * scale)
        cv2.circle(vis, (disp_x, disp_y), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(idx+1), (disp_x+5, disp_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if len(points) == 4:
        disp_pts = np.array([(int(pt[0]*scale), int(pt[1]*scale)) for pt in points])
        cv2.polylines(vis, [disp_pts], isClosed=True, color=(255,0,0), thickness=2)
    cv2.imshow('Select 4 Points', vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        points = []
        print("Points reset.")
    elif key == ord('q') and len(points) == 4:
        break

cv2.destroyAllWindows()

print("\nSelected points (in order):")
for idx, pt in enumerate(points):
    print(f"Point {idx+1}: {pt}")

# Save points to file
np.save('bev_src_points.npy', np.array(points))
print("\nPoints saved to bev_src_points.npy") 