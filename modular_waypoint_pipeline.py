# modular_waypoint_pipeline.py
import os
import cv2
import math
import numpy as np
import networkx as nx

# ---- Your detector ----
from fast_road_detector import FastRoadDetector, Config

# =============================================================================
# Configuration
# =============================================================================
ROAD_ID = 1
SIDEWALK_ID = 2

src_points = np.array([
    [0.0,   717.0],
    [1278.0, 717.0],
    [860.0,  337.0],
    [573.0,  329.0]
], dtype=np.float32)

dst_points = np.array([
    [100, 480],  # bottom-left
    [500, 480],  # bottom-right
    [400, 100],  # top-right
    [200, 100]   # top-left
], dtype=np.float32)

bev_size = (600, 500)  # (W, H)
H = cv2.getPerspectiveTransform(src_points, dst_points)
Hinv = np.linalg.inv(H)

TRIM_BOTTOM = 20

PATH_COLORS = [
    (0,255,255), (255,255,0), (255,0,255),
    (0,165,255), (0,255,128), (128,0,255),
    (255,128,0), (0,128,255), (128,255,0)
]

# =============================================================================
# Utilities
# =============================================================================
def split_masks_from_output(model_output, road_id=ROAD_ID, sidewalk_id=SIDEWALK_ID):
    m = model_output.astype(np.uint8) if model_output.dtype != np.uint8 else model_output
    uniq = set(np.unique(m).tolist())
    if uniq.issubset({0, 255}):
        sidewalk = (m > 0).astype(np.uint8) * 255
        road = np.zeros_like(sidewalk, dtype=np.uint8)
        return sidewalk, road
    sidewalk = (m == sidewalk_id).astype(np.uint8) * 255
    road     = (m == road_id).astype(np.uint8) * 255
    return sidewalk, road

def colorize_sidewalk_road(frame_bgr, sidewalk_mask_255, road_mask_255, alpha=0.45):
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    color_layer[road_mask_255 > 0]     = (255, 120, 0)
    color_layer[sidewalk_mask_255 > 0] = (0, 200, 0)
    cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

# =============================================================================
# Skeletonization (Zhang–Suen)
# =============================================================================
def zhang_suen_thinning(bin_img_0_255):
    img = (bin_img_0_255 > 0).astype(np.uint8)
    h, w = img.shape

    def neighbors(y, x):
        return [img[y-1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1],
                img[y+1, x], img[y+1, x-1], img[y, x-1], img[y-1, x-1]]

    def transitions(nb):
        return sum((nb[i] == 0 and nb[(i+1) % 8] == 1) for i in range(8))

    while True:
        changing1 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if img[y, x] != 1: continue
                nb = neighbors(y, x); C = transitions(nb); N = sum(nb)
                if (2 <= N <= 6 and C == 1 and nb[0]*nb[2]*nb[4] == 0 and nb[2]*nb[4]*nb[6] == 0):
                    changing1.append((y, x))
        for y, x in changing1: img[y, x] = 0

        changing2 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if img[y, x] != 1: continue
                nb = neighbors(y, x); C = transitions(nb); N = sum(nb)
                if (2 <= N <= 6 and C == 1 and nb[0]*nb[2]*nb[6] == 0 and nb[0]*nb[4]*nb[6] == 0):
                    changing2.append((y, x))
        for y, x in changing2: img[y, x] = 0

        if not changing1 and not changing2: break

    return (img * 255).astype(np.uint8)

# =============================================================================
# BEV skeleton & graph
# =============================================================================
def extract_skeleton_graph(bev_binary_0_255, trim_px=5):
    kernel = np.ones((5, 5), np.uint8)
    bev_clean = cv2.morphologyEx(bev_binary_0_255, cv2.MORPH_CLOSE, kernel)
    bev_clean = cv2.medianBlur(bev_clean, 5)
    _, binary = cv2.threshold(bev_clean, 127, 255, cv2.THRESH_BINARY)

    skeleton = zhang_suen_thinning(binary)
    sk = skeleton.copy()

    if trim_px > 0:
        sk[:trim_px, :]  = 0
        sk[:, :trim_px]  = 0
        sk[:, -trim_px:] = 0
        sk[-trim_px:, :] = 0

    G = nx.Graph()
    h, w = sk.shape
    for y in range(h):
        xs = np.where(sk[y] == 255)[0]
        for x in xs:
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx == 0 and dy == 0: continue
                    ny, nx_ = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx_ < w and sk[ny,nx_] == 255:
                        G.add_edge((x,y),(nx_,ny),weight=math.hypot(dx,dy))
    return sk, G

def skeleton_endpoints(G):
    return [n for n in G.nodes if G.degree[n] == 1]

# =============================================================================
# Model init
# =============================================================================
def initialize_model():
    cfg = Config(model_dir="models/my-segformer-road", conf_thresh=0.5, road_id=ROAD_ID)
    return FastRoadDetector(cfg)

# =============================================================================
# Main
# =============================================================================
def process_video(video_path, output_dir, stride=1, save_video=False):
    print("🔧 Initializing FastRoadDetector...")
    model = initialize_model(); print("✅ Model ready!")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    vw = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(os.path.join(output_dir,"cam_paths.mp4"),fourcc,fps,(w,h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_id % stride != 0:
            frame_id += 1; continue

        model_out,_ = model.process_frame(frame)
        if model_out.shape != frame.shape[:2]:
            model_out = cv2.resize(model_out,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
        sidewalk_mask, road_mask = split_masks_from_output(model_out)

        cam_overlay = colorize_sidewalk_road(frame, sidewalk_mask, road_mask)

        bev_sidewalk = cv2.warpPerspective(sidewalk_mask, H, bev_size)
        bev_road     = cv2.warpPerspective(road_mask, H, bev_size)

        if TRIM_BOTTOM > 0:
            bev_sidewalk = bev_sidewalk[:bev_sidewalk.shape[0]-TRIM_BOTTOM, :]
            bev_road     = bev_road[:bev_road.shape[0]-TRIM_BOTTOM, :]

        skeleton_mask, graph = extract_skeleton_graph(bev_sidewalk, trim_px=5)
        endpoints = skeleton_endpoints(graph)

        # Pick the endpoint with largest y (lowest point)
        start = None
        if endpoints:
            start = max(endpoints, key=lambda p: p[1])

        H_bev,W_bev = skeleton_mask.shape
        bev_color = np.zeros((H_bev,W_bev,3),dtype=np.uint8)
        bev_color[bev_road>0]=(255,120,0)
        bev_color[bev_sidewalk>0]=(0,200,0)

        # 🔹 Thicken skeleton for visualization only
        vis_skel = cv2.dilate(
            skeleton_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35)),
            iterations=1
        )
        gy, gx = np.where(vis_skel > 0)
        bev_color[gy, gx] = (0,128,0)

        cam_paths = cam_overlay.copy()

        if start:
            other_endpoints = [e for e in endpoints if e != start]
            for idx, end in enumerate(other_endpoints):
                try:
                    path = nx.dijkstra_path(graph, start, end, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                color = PATH_COLORS[idx % len(PATH_COLORS)]

                # --- draw path on BEV as wide ribbon ---
                path_np = np.int32(path).reshape(-1,1,2)
                cv2.polylines(bev_color, [path_np], False, color, thickness=25, lineType=cv2.LINE_AA)

                # endpoints
                cv2.circle(bev_color,(int(start[0]),int(start[1])),8,(0,0,255),-1)
                cv2.circle(bev_color,(int(end[0]),int(end[1])),6,color,-1)

                # --- draw path on camera by warping ribbon ---
                ribbon_mask = np.zeros((H_bev, W_bev), np.uint8)
                cv2.polylines(ribbon_mask, [path_np], False, 255, thickness=35, lineType=cv2.LINE_AA)

                cam_ribbon = cv2.warpPerspective(ribbon_mask, Hinv, (frame.shape[1], frame.shape[0]))
                color_layer = np.zeros_like(frame)
                color_layer[cam_ribbon > 0] = color
                cam_paths = cv2.addWeighted(color_layer, 0.6, cam_paths, 1.0, 0)

        cv2.imwrite(os.path.join(output_dir,f"bev_paths_{frame_id:04d}.png"),bev_color)
        cv2.imwrite(os.path.join(output_dir,f"cam_paths_{frame_id:04d}.png"),cam_paths)
        if vw: vw.write(cam_paths)

        print(f"Processed frame {frame_id} | start={start} | num_paths={len(other_endpoints)}")
        frame_id += 1

    cap.release()
    if vw: vw.release()
    print("✅ Done. Results saved to", output_dir)

# =============================================================================
# Entry
# =============================================================================
if __name__=="__main__":
    process_video("test_video_june_03_3.mp4","bev_paths_dijkstra",stride=3,save_video=False)
