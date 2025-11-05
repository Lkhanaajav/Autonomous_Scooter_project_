# -*- coding: utf-8 -*-
# waypoint_skeleton_pipeline_v5.py
# Segmentation → BEV → Distance-based BEV cleaning (removes thin branches)
# → Guo–Hall Skeleton → Pruned Main Path → Waypoints → Direction Commands

import os, cv2, time, math
import numpy as np
from collections import defaultdict
from fast_road_detector import FastRoadDetector, Config
import networkx as nx

# ========= Settings =========
ROAD_ID = 1
SIDEWALK_ID = 2
MODEL_DIR = "models/my-segformer-road"
CONF_THRESH = 0.5
INPUT = "test_video_june_03_3.mp4"   # or int(0) for webcam
STRIDE = 5
SHOW_SEG = True
SAVE_VIDEO = True
# SegFormer inference overrides / benchmarking
SEG_INPUT_RES = (640, 360)  # (width, height) or None to use processor default
SEG_BENCHMARK_RESOLUTIONS = [
    (640, 360),
]  # resolution specs to benchmark before streaming
SEG_BENCHMARK_WARMUP = 1
SEG_BENCHMARK_REPEATS = 5
SEG_BENCHMARK_VISUALIZE = True
SEG_BENCHMARK_SAVE = True
SEG_BENCHMARK_SAVE_DIR = "benchmarks/segformer_samples"
SEG_BENCHMARK_PREVIEW_WIDTH = 960  # resize width for comparison panels; set 0 to disable resize
SEG_BENCHMARK_SHOW_MASK = True
SEG_BENCHMARK_GRID_COLS = 2
SEG_BENCHMARK_PANEL_GAP = 18
SEG_BENCHMARK_LABEL_FONT_SCALE = 1.3
SEG_BENCHMARK_LABEL_THICKNESS = 3
SEG_BENCHMARK_LABEL_PAD = 8
SEG_BENCHMARK_LABEL_BG = (255,255,255)
SEG_BENCHMARK_LABEL_COLOR = (0,0,0)
     # (W, H)
TRIM_BOTTOM = 30
CAM_FPS_HINT = 30

# Tuning knobs
SCALE_SKELETON = 0.5
PRUNE_BRANCH_LEN = 12
WAYPOINT_ARCSTEP = 40.0
WAYPOINT_COUNT_MAX = 12
EMA_ALPHA = 0.35
DT_CORE_THRESH = 6.0
TURN_THRESH_DEG = 20.0       # angle threshold for left/right turns
# ============================

# Output video path for demo (side-by-side Cam + BEV)
DEMO_VIDEO_PATH = "demo_overlay.mp4"

# Visualization palette for multi-path ribbons
PATH_COLORS = [
    (0,255,255), (255,255,0), (255,0,255),
    (0,165,255), (0,255,128), (128,0,255),
    (255,128,0), (0,128,255), (128,255,0)
]

# Bottom band (in pixels) used to pick start seeds on the BEV skeleton
BOTTOM_BAND_PX = 30

# Optional: straighten BEV boundaries by column-wise smoothing
STRAIGHTEN_EDGES = False
EDGE_SMOOTH_WIN = 21         # odd window size in pixels
EDGE_MIN_WIDTH_PX = 8        # ignore columns narrower than this width
KEEP_MAIN_COMPONENT = True
MAIN_COMPONENT_BOTTOM_BAND = 45
MAIN_COMPONENT_CENTER_WEIGHT = 0.35

src_points = np.array([[4, 716], [1275, 716], [777, 291], [654, 289]], dtype=np.float32)  # [BL, BR, TR, TL]

dst_points = np.array([
    [100, 880],  # bottom-left (close to camera)
    [500, 880],  # bottom-right
    [400, 60],   # top-right (farther upward)
    [200, 60]    # top-left
], dtype=np.float32)

BEV_SIZE = (600, 900) 



H = cv2.getPerspectiveTransform(src_points, dst_points)
Hinv = np.linalg.inv(H)

# ========= Timing =========
class TimingAnalyzer:
    def __init__(self):
        self.timings = defaultdict(list)
        self.tick = {}
    def start(self, name): self.tick[name] = time.time()
    def end(self, name):
        t0 = self.tick.pop(name, None)
        if t0 is not None: self.timings[name].append(time.time() - t0)
    def summary_str(self):
        out=[]
        for k,v in self.timings.items():
            if v: out.append(f"{k:15s} avg {np.mean(v)*1000:6.1f} ms  count {len(v)}")
        return "\n".join(out)
timing = TimingAnalyzer()

# ========= Helpers =========
def split_masks_from_output(m, road_id=ROAD_ID, sidewalk_id=SIDEWALK_ID):
    m = m.astype(np.uint8)
    if set(np.unique(m)) <= {0,255}:
        sidewalk = (m>0).astype(np.uint8)*255
        road = np.zeros_like(sidewalk)
    else:
        sidewalk = (m==sidewalk_id).astype(np.uint8)*255
        road = (m==road_id).astype(np.uint8)*255
    return sidewalk, road

# ========= Benchmarking =========
def _resolve_resolution_spec(spec):
    if spec is None:
        return None, "native", True
    if isinstance(spec, str):
        text = spec.strip().lower()
        if text in {"native", "default", "none"}:
            return None, "native", True
        if "x" in text:
            parts = text.replace(" ", "").split("x")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                return (w, h), f"{w}x{h}", False
        if text.isdigit():
            val = int(text)
            return (val, val), f"{val}x{val}", False
        raise ValueError(f"Cannot parse resolution spec: {spec}")
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        w, h = int(spec[0]), int(spec[1])
        return (w, h), f"{w}x{h}", False
    if isinstance(spec, int):
        val = int(spec)
        return (val, val), f"{val}x{val}", False
    raise ValueError(f"Unsupported resolution spec: {spec}")


def benchmark_segformer_resolutions(detector, frame, specs, warmup=1, repeats=5):
    if not specs:
        return
    prev_mask = detector.previous_mask.copy() if detector.previous_mask is not None else None
    prev_processed = detector.performance_metrics.processed_count
    prev_inference = detector.performance_metrics.inference_time
    original_resize = detector.config.inference_resize
    viz_entries = []
    print("\n=== SegFormer Input Resolution Benchmark ===")
    try:
        for spec in specs:
            size_override, label, use_native = _resolve_resolution_spec(spec)
            detector.previous_mask = prev_mask if prev_mask is None else prev_mask.copy()
            detector.performance_metrics.processed_count = prev_processed
            detector.performance_metrics.inference_time = prev_inference
            if use_native:
                detector.config.inference_resize = None
                override_arg = None
            else:
                detector.config.inference_resize = original_resize
                override_arg = size_override
            warmup_count = max(0, int(warmup))
            repeat_count = max(1, int(repeats))
            for _ in range(warmup_count):
                detector.process_frame(frame, processor_size=override_arg)
            timings = []
            for _ in range(repeat_count):
                t0 = time.perf_counter()
                detector.process_frame(frame, processor_size=override_arg)
                timings.append((time.perf_counter() - t0) * 1000.0)
            mean_ms = float(np.mean(timings)) if timings else float("nan")
            std_ms = float(np.std(timings)) if len(timings) > 1 else 0.0
            viz_mask, viz_overlay = detector.process_frame(frame, processor_size=override_arg)
            detector.previous_mask = prev_mask if prev_mask is None else prev_mask.copy()
            detector.performance_metrics.processed_count = prev_processed
            detector.performance_metrics.inference_time = prev_inference
            print(f"  {label:>10s} : {mean_ms:7.2f} ms  (± {std_ms:5.2f})")
            if SEG_BENCHMARK_VISUALIZE or SEG_BENCHMARK_SAVE:
                viz_entries.append({
                    "label": label,
                    "mean": mean_ms,
                    "std": std_ms,
                    "mask": viz_mask.copy(),
                    "overlay": viz_overlay.copy(),
                })
    finally:
        detector.config.inference_resize = original_resize
        detector.previous_mask = prev_mask if prev_mask is None else prev_mask.copy()
        detector.performance_metrics.processed_count = prev_processed
        detector.performance_metrics.inference_time = prev_inference
    print("===========================================\n")
    if viz_entries and (SEG_BENCHMARK_VISUALIZE or SEG_BENCHMARK_SAVE):
        panels=[]
        for entry in viz_entries:
            panel = entry["overlay"].copy()
            if SEG_BENCHMARK_SHOW_MASK:
                mask_color = cv2.cvtColor(entry["mask"], cv2.COLOR_GRAY2BGR)
                panel = np.hstack((panel, mask_color))
            text = f"{entry['label']} | {entry['mean']:.1f} ms"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = float(SEG_BENCHMARK_LABEL_FONT_SCALE)
            thickness = int(max(1, SEG_BENCHMARK_LABEL_THICKNESS))
            pad = int(max(0, SEG_BENCHMARK_LABEL_PAD))
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            x_text = pad
            y_text = pad + th
            x1 = max(0, x_text - pad)
            y1 = max(0, y_text - th - pad)
            x2 = min(panel.shape[1], x_text + tw + pad)
            y2 = min(panel.shape[0], y_text + baseline + pad)
            cv2.rectangle(panel, (x1, y1), (x2, y2), SEG_BENCHMARK_LABEL_BG, -1)
            cv2.putText(panel, text, (x_text, y_text), font, scale, SEG_BENCHMARK_LABEL_COLOR, thickness, cv2.LINE_AA)
            if SEG_BENCHMARK_PREVIEW_WIDTH:
                target_w = int(SEG_BENCHMARK_PREVIEW_WIDTH)
                if target_w > 0 and panel.shape[1] != target_w:
                    scale = target_w / float(panel.shape[1])
                    target_h = max(1, int(round(panel.shape[0] * scale)))
                    panel = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)
            panels.append(panel)
        if panels:
            widths = [p.shape[1] for p in panels]
            heights = [p.shape[0] for p in panels]
            max_w = max(widths)
            max_h = max(heights)
            padded = []
            for p in panels:
                h_pad = max_h - p.shape[0]
                w_pad = max_w - p.shape[1]
                if w_pad > 0:
                    pad = np.zeros((p.shape[0], w_pad, 3), dtype=np.uint8)
                    p = np.hstack((p, pad))
                if h_pad > 0:
                    pad = np.zeros((h_pad, p.shape[1], 3), dtype=np.uint8)
                    p = np.vstack((p, pad))
                padded.append(p)

            cols = SEG_BENCHMARK_GRID_COLS if SEG_BENCHMARK_GRID_COLS else 1
            cols = max(1, int(cols))
            gap = max(0, int(SEG_BENCHMARK_PANEL_GAP))
            rows = int(math.ceil(len(padded) / float(cols)))
            blank = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            row_imgs = []
            for r in range(rows):
                chunk = padded[r*cols:(r+1)*cols]
                while len(chunk) < cols:
                    chunk.append(blank.copy())
                if gap > 0:
                    gap_col = np.zeros((max_h, gap, 3), dtype=np.uint8)
                    row_img = chunk[0]
                    for cimg in chunk[1:]:
                        row_img = np.hstack((row_img, gap_col, cimg))
                else:
                    row_img = np.hstack(chunk)
                row_imgs.append(row_img)

            if gap > 0:
                gap_row = np.zeros((gap, row_imgs[0].shape[1], 3), dtype=np.uint8)
                viz_canvas = row_imgs[0]
                for rimg in row_imgs[1:]:
                    viz_canvas = np.vstack((viz_canvas, gap_row, rimg))
            else:
                viz_canvas = np.vstack(row_imgs)
            if SEG_BENCHMARK_VISUALIZE:
                cv2.imshow("SegFormer Resolution Comparison", viz_canvas)
                cv2.waitKey(1)
            if SEG_BENCHMARK_SAVE:
                os.makedirs(SEG_BENCHMARK_SAVE_DIR, exist_ok=True)
                fname = time.strftime("segformer_compare_%Y%m%d_%H%M%S.png")
                cv2.imwrite(os.path.join(SEG_BENCHMARK_SAVE_DIR, fname), viz_canvas)

# ========= BEV cleaning =========
def clean_sidewalk_mask(bev_mask_255, dt_thresh=DT_CORE_THRESH):
    mask = bev_mask_255.copy().astype(np.uint8)
    k7 = np.ones((7,7), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)
    dist = cv2.distanceTransform((mask>0).astype(np.uint8), cv2.DIST_L2, 5)
    core = (dist > float(dt_thresh)).astype(np.uint8) * 255
    core = cv2.dilate(core, k5, iterations=1)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k5, iterations=1)
    return core

def straighten_bev_edges(mask_255, win=EDGE_SMOOTH_WIN, min_width_px=EDGE_MIN_WIDTH_PX):
    """
    Column-wise boundary smoothing in BEV to produce straighter edges.
    For each x, find top/bottom y of foreground; smooth with moving average; refill.
    """
    m = (mask_255 > 0).astype(np.uint8)
    H, W = m.shape
    top = np.full(W, -1, dtype=np.int32)
    bot = np.full(W, -1, dtype=np.int32)
    last_top = -1
    last_bot = -1
    for x in range(W):
        ys = np.where(m[:, x] > 0)[0]
        if ys.size >= min_width_px:
            top[x] = int(ys.min())
            bot[x] = int(ys.max())
            last_top = top[x]
            last_bot = bot[x]
        else:
            # propagate last seen to avoid gaps
            top[x] = last_top
            bot[x] = last_bot

    # replace remaining -1 by nearest valid values
    for arr in (top, bot):
        # forward fill
        for x in range(1, W):
            if arr[x] < 0:
                arr[x] = arr[x-1]
        # backward fill
        for x in range(W-2, -1, -1):
            if arr[x] < 0:
                arr[x] = arr[x+1]

    if win % 2 == 0:
        win += 1
    if win < 3:
        win = 3
    k = np.ones(win, dtype=np.float32) / float(win)
    # pad by reflection for convolution
    def smooth_1d(a):
        pad = win // 2
        ap = np.pad(a.astype(np.float32), (pad, pad), mode='reflect')
        s = np.convolve(ap, k, mode='valid')
        return s.astype(np.float32)

    top_s = smooth_1d(top)
    bot_s = smooth_1d(bot)

    out = np.zeros_like(m, dtype=np.uint8)
    for x in range(W):
        y1 = int(round(top_s[x] if np.isfinite(top_s[x]) else 0))
        y2 = int(round(bot_s[x] if np.isfinite(bot_s[x]) else -1))
        y1 = max(0, min(H-1, y1))
        y2 = max(0, min(H-1, y2))
        if y2 >= y1:
            out[y1:y2+1, x] = 255
    return out


def select_main_component(mask_255, bottom_band_px=MAIN_COMPONENT_BOTTOM_BAND, center_weight=MAIN_COMPONENT_CENTER_WEIGHT):
    bin_ = (mask_255 > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    if num <= 1:
        return mask_255
    H, W = mask_255.shape
    bottom_band_px = int(max(1, min(H, bottom_band_px)))
    bottom_slice = labels[max(0, H - bottom_band_px):, :]
    center_x = W / 2.0
    best_label = None
    best_score = -1.0
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        touches_bottom = bool(np.any(bottom_slice == idx))
        cx, _ = centroids[idx]
        center_bonus = 1.0 - min(1.0, abs(cx - center_x) / (center_x + 1e-6))
        score = area
        if touches_bottom:
            score += area * 0.5
        score += area * float(center_weight) * max(0.0, center_bonus)
        if score > best_score:
            best_score = score
            best_label = idx
    if best_label is None:
        best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask_255, dtype=np.uint8)
    out[labels == best_label] = 255
    return out

# ========= Skeletonization =========
def skeletonize_cv2(mask_255):
    """Fast & accurate skeletonization using OpenCV Guo–Hall."""
    from cv2.ximgproc import thinning, THINNING_GUOHALL
    bin_ = ((mask_255 > 0).astype(np.uint8)) * 255
    skel = thinning(bin_, THINNING_GUOHALL)
    skel = (skel * 255).astype(np.uint8)
    return skel

def extract_skeleton(bev_binary_0_255, trim_px=5):
    kernel = np.ones((5,5), np.uint8)
    bev_clean = cv2.morphologyEx(bev_binary_0_255, cv2.MORPH_CLOSE, kernel)
    bev_clean = cv2.medianBlur(bev_clean,5)
    _, binary = cv2.threshold(bev_clean,127,255,cv2.THRESH_BINARY)
    skeleton = skeletonize_cv2(binary)
    if trim_px>0:
        skeleton[:trim_px,:]=0
        skeleton[-trim_px:,:]=0
        skeleton[:, :trim_px]=0
        skeleton[:, -trim_px:]=0
    return skeleton

# ========= Skeleton utilities =========
def prune_small_branches(skel, min_len=PRUNE_BRANCH_LEN):
    s = skel.copy()
    for _ in range(min_len):
        nb = cv2.filter2D((s>0).astype(np.uint8), -1, np.ones((3,3), np.uint8))
        endpoints = ((s>0) & (nb==2))
        s[endpoints] = 0
    return s

def main_path_from_bottom_center(skel):
    h, w = skel.shape
    bin_ = (skel > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_)
    if num <= 1:
        return skel
    x_center = w // 2
    zone_w = int(w * 0.40)
    zone_h = int(h * 0.15)
    x1, x2 = x_center - zone_w//2, x_center + zone_w//2
    y1, y2 = h - zone_h, h
    main_label, best_area = None, 0
    for cid in range(1, num):
        comp_mask = (labels == cid)
        if np.any(comp_mask[y1:y2, x1:x2]):
            area = stats[cid, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                main_label = cid
    if main_label is None:
        main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main = ((labels == main_label).astype(np.uint8) * 255)
    return main

def order_path_pixels_by_y_then_x(skel):
    ys, xs = np.where(skel>0)
    if len(xs)==0: return []
    pts = np.stack((xs, ys), axis=1)
    xs_center = np.median(xs)
    sort_key = np.lexsort((np.abs(pts[:,0]-xs_center), pts[:,1]))
    return [tuple(p) for p in pts[sort_key]]

def sample_by_arclength(path_xy, step=WAYPOINT_ARCSTEP, max_pts=WAYPOINT_COUNT_MAX):
    if not path_xy: return []
    keep=[path_xy[0]]; acc=0.0
    for i in range(1,len(path_xy)):
        acc += math.hypot(path_xy[i][0]-path_xy[i-1][0], path_xy[i][1]-path_xy[i-1][1])
        if acc >= step:
            keep.append(path_xy[i]); acc = 0.0
            if len(keep) >= max_pts: break
    if len(keep)<max_pts and path_xy[-1]!=keep[-1]:
        keep.append(path_xy[-1])
    return keep

def ema_smooth(prev_pts, new_pts, alpha=EMA_ALPHA):
    if not prev_pts or not new_pts or len(prev_pts)!=len(new_pts): return new_pts
    out=[]
    for (px,py),(nx,ny) in zip(prev_pts,new_pts):
        sx = int(round(alpha*nx + (1-alpha)*px))
        sy = int(round(alpha*ny + (1-alpha)*py))
        out.append((sx,sy))
    return out

# ========= Command generation =========
# def compute_commands(waypoints, turn_thresh_deg=TURN_THRESH_DEG):
#     """
#     Given ordered waypoints [(x,y), ...], compute simple textual commands.
#     Returns list like ['STRAIGHT', 'LEFT', 'RIGHT', 'STRAIGHT', ...]
#     """
#     if len(waypoints) < 3:
#         return []
#     cmds = []
#     def heading(p1, p2):
#         return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
#     for i in range(1, len(waypoints)-1):
#         h1 = heading(waypoints[i-1], waypoints[i])
#         h2 = heading(waypoints[i], waypoints[i+1])
#         dtheta = (h2 - h1 + 180) % 360 - 180
#         if abs(dtheta) < turn_thresh_deg:
#             cmds.append("STRAIGHT")
#         elif dtheta > 0:
#             cmds.append("TURN LEFT")
#         else:
#             cmds.append("TURN RIGHT")
#     cmds.append("STOP")
#     return cmds

# ========= HUD =========
def put_hud(img, lines, org=(10,25), scale=0.6, color=(255,255,255)):
    y=org[1]
    for text in lines:
        cv2.putText(img,text,(org[0],y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)
        y+=int(24*scale)

def put_hud_panel(img, lines, org=(10, 25), scale=0.7):
    # Semi-transparent panel under text
    pad_x, pad_y = 10, 8
    max_w = 0
    h_line = int(24 * scale)
    for t in lines:
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        max_w = max(max_w, tw)
    panel_w = max_w + 2 * pad_x
    panel_h = h_line * len(lines) + 2 * pad_y
    x0, y0 = org[0], org[1] - int(0.8 * h_line)
    x1, y1 = x0 + panel_w, y0 + panel_h
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    # Draw text on top
    y = org[1]
    for text in lines:
        cv2.putText(img, text, (org[0] + pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += h_line

# ========= Main =========
def main():
    cfg_resize = None
    if SEG_INPUT_RES is not None:
        size_override, _, use_native = _resolve_resolution_spec(SEG_INPUT_RES)
        cfg_resize = None if use_native else size_override
    cfg=Config(model_dir=MODEL_DIR, conf_thresh=CONF_THRESH, road_id=ROAD_ID, inference_resize=cfg_resize)
    model=FastRoadDetector(cfg)
    cap=cv2.VideoCapture(INPUT if not str(INPUT).isdigit() else int(INPUT))
    if not cap.isOpened():
        print(f"ERROR: cannot open {INPUT}"); return
    fps_in=cap.get(cv2.CAP_PROP_FPS) or CAM_FPS_HINT
    frame_id=0; times=[]; last_mask=None; last_waypoints_cam=None; did_benchmark=False

    # Lazy-init video writer after first composed frame (to know exact size)
    vw_demo=None

    print("Running waypoint pipeline… (ESC to quit)")

    while True:
        t0=time.time()
        ok,frame=cap.read()
        if not ok: break
        run_net=(frame_id%STRIDE==0)

        if SEG_BENCHMARK_RESOLUTIONS and not did_benchmark:
            benchmark_segformer_resolutions(
                model,
                frame,
                SEG_BENCHMARK_RESOLUTIONS,
                warmup=SEG_BENCHMARK_WARMUP,
                repeats=SEG_BENCHMARK_REPEATS
            )
            did_benchmark=True

        # 1) Inference
        timing.start("inference")
        if run_net or last_mask is None:
            seg,_=model.process_frame(frame)
            if seg.shape!=frame.shape[:2]:
                seg=cv2.resize(seg,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
            last_mask=seg
        seg=last_mask
        timing.end("inference")

        # 2) Split mask
        sidewalk_mask,road_mask=split_masks_from_output(seg)

        # 3) BEV + cleaning
        timing.start("bev")
        bev_sidewalk=cv2.warpPerspective(sidewalk_mask,H,BEV_SIZE)
        bev_road=cv2.warpPerspective(road_mask,H,BEV_SIZE)
        if TRIM_BOTTOM>0: bev_sidewalk=bev_sidewalk[:-TRIM_BOTTOM,:]
        if TRIM_BOTTOM>0: bev_road=bev_road[:-TRIM_BOTTOM,:]
        bev_sidewalk = clean_sidewalk_mask(bev_sidewalk, DT_CORE_THRESH)
        if STRAIGHTEN_EDGES:
            bev_sidewalk = straighten_bev_edges(bev_sidewalk, EDGE_SMOOTH_WIN, EDGE_MIN_WIDTH_PX)
        if KEEP_MAIN_COMPONENT:
            bev_sidewalk = select_main_component(
                bev_sidewalk,
                bottom_band_px=MAIN_COMPONENT_BOTTOM_BAND,
                center_weight=MAIN_COMPONENT_CENTER_WEIGHT,
            )
        timing.end("bev")

        # 4) Skeleton
        timing.start("skeleton")
        skel = extract_skeleton(bev_sidewalk, trim_px=5)
        skel = prune_small_branches(skel, PRUNE_BRANCH_LEN)
        main_skel = main_path_from_bottom_center(skel)
        timing.end("skeleton")

        # 5) Waypoints (from main skeleton only)
        timing.start("waypoints")
        ordered = order_path_pixels_by_y_then_x(main_skel)
        waypoints_bev = sample_by_arclength(ordered, WAYPOINT_ARCSTEP, WAYPOINT_COUNT_MAX)

        if waypoints_bev:
            pts = np.array(waypoints_bev, dtype=np.float32).reshape(-1,1,2)
            cam_pts = cv2.perspectiveTransform(pts, Hinv).reshape(-1,2)
            waypoints_cam = [tuple(map(int,p)) for p in cam_pts]
            waypoints_cam = ema_smooth(last_waypoints_cam, waypoints_cam, EMA_ALPHA)
            last_waypoints_cam = waypoints_cam
        else:
            waypoints_cam = last_waypoints_cam or []
        timing.end("waypoints")

        # 6) Dijkstra on full skeleton starting from bottom band, enumerate all reachable endpoints
        # Build graph from skel pixels (8-connectivity)
        Hm, Wm = skel.shape
        G = nx.Graph()
        for y in range(Hm):
            xs = np.where(skel[y] > 0)[0]
            for x in xs:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < Hm and 0 <= nx_ < Wm and skel[ny, nx_] > 0:
                            G.add_edge((x, y), (nx_, ny), weight=math.hypot(dx, dy))

        paths = []  # list of (path_pts, length)
        if G.number_of_nodes() > 1:
            endpoints = [n for n in G.nodes if G.degree[n] == 1]
            # Choose start from bottom band, prefer endpoints; break ties by closeness to center
            center_x = Wm // 2
            band_nodes = [n for n in G.nodes if n[1] >= Hm - BOTTOM_BAND_PX]
            band_endpoints = [n for n in endpoints if n[1] >= Hm - BOTTOM_BAND_PX]
            def start_key(p):
                return (p[1], -abs(p[0] - center_x))  # max y, then nearest to center
            if band_endpoints:
                start = max(band_endpoints, key=start_key)
            elif band_nodes:
                start = max(band_nodes, key=start_key)
            else:
                # fallback: global lowest endpoint or node
                start = max(endpoints, key=lambda p: p[1]) if endpoints else max(G.nodes, key=lambda p: p[1])

            # Dijkstra to all other endpoints
            for end in endpoints:
                if end == start:
                    continue
                try:
                    path = nx.dijkstra_path(G, start, end, weight="weight")
                    plen = nx.path_weight(G, path, weight="weight")
                    paths.append((path, plen))
                except nx.NetworkXNoPath:
                    continue

            # Sort by length (longest first)
            paths.sort(key=lambda x: x[1], reverse=True)

        # 6) Commands (disabled for demo-only video)
        # commands = compute_commands(waypoints_bev)
        # if commands:
        #     print(" → ".join(commands))
        #     current_cmd = commands[0]
        # else:
        #     current_cmd = "..."
        
        # 7) Visualization (nice skeleton styling)
        timing.start("viz")
        H_bev, W_bev = bev_sidewalk.shape
        bev_rgb=np.zeros((H_bev,W_bev,3),np.uint8)

        # Base color layers
        bev_rgb[bev_road>0]=(0,128,255)       # road: blue-orange tone
        bev_rgb[bev_sidewalk>0]=(120,255,120) # sidewalk: soft green

        # Smooth, thick skeleton overlay
        vis_skel = cv2.dilate(
            skel,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            iterations=1
        )
        skeleton_layer = np.zeros_like(bev_rgb)
        skeleton_layer[vis_skel > 0] = (100, 220, 255)  # sky blue
        bev_rgb = cv2.addWeighted(skeleton_layer, 0.7, bev_rgb, 1.0, 0)

        # Highlight main skeleton stronger
        vis_main = cv2.dilate(
            main_skel,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
            iterations=1
        )
        gy, gx = np.where(vis_main > 0)
        bev_rgb[gy, gx] = (0,128,0)

        # Subtle contrast boost for paper/demo clarity
        bev_rgb = cv2.GaussianBlur(bev_rgb, (3, 3), 0)
        bev_rgb = cv2.addWeighted(bev_rgb, 1.2, bev_rgb, 0, 0)

        cam=frame.copy()
        if SHOW_SEG:
            overlay=np.zeros_like(cam)
            overlay[road_mask>0]=(0,140,255)
            overlay[sidewalk_mask>0]=(0,200,0)
            cam=cv2.addWeighted(overlay,0.35,cam,1.0,0)

        # Draw all available paths (from start to each endpoint) as ribbons on BEV and Cam
        for idx, (path_pts, plen) in enumerate(paths):
            color = PATH_COLORS[idx % len(PATH_COLORS)]
            path_np = np.int32(path_pts).reshape(-1, 1, 2)
            # BEV ribbon
            ribbon_mask = np.zeros((H_bev, W_bev), np.uint8)
            cv2.polylines(ribbon_mask, [path_np], False, 255, thickness=28, lineType=cv2.LINE_AA)
            ribbon_layer = np.zeros_like(bev_rgb)
            ribbon_layer[ribbon_mask > 0] = color
            glow = cv2.GaussianBlur(ribbon_layer, (0, 0), 6)
            bev_rgb = cv2.addWeighted(glow, 0.22, bev_rgb, 1.0, 0)
            bev_rgb = cv2.addWeighted(ribbon_layer, 0.65, bev_rgb, 1.0, 0)

            # Cam ribbon via inverse homography
            cam_ribbon = cv2.warpPerspective(ribbon_mask, Hinv, (cam.shape[1], cam.shape[0]))
            cam_color = np.zeros_like(cam)
            cam_color[cam_ribbon > 0] = color
            cam_glow = cv2.GaussianBlur(cam_color, (0, 0), 6)
            cam = cv2.addWeighted(cam_glow, 0.22, cam, 1.0, 0)
            cam = cv2.addWeighted(cam_color, 0.55, cam, 1.0, 0)

        dt=time.time()-t0
        times.append(dt)
        fps=1.0/dt if dt>0 else 0
        hud=[
            f"Frame {frame_id}",
            f"FPS {fps:4.1f}  (dt {(dt*1000):.1f} ms)",
            f"Infer  {(timing.timings['inference'][-1]*1000 if timing.timings['inference'] else 0):.1f} ms",
            f"BEV    {(timing.timings['bev'][-1]*1000 if timing.timings['bev'] else 0):.1f} ms",
            f"Skel   {(timing.timings['skeleton'][-1]*1000 if timing.timings['skeleton'] else 0):.1f} ms",
            f"Paths  {len(paths)}",
        ]
        put_hud(cam,hud,org=(10,28),scale=0.7)

        cv2.imshow("Cam",cam)
        cv2.imshow("BEV",bev_rgb)

        # Compose side-by-side demo frame and write to video if enabled
        if SAVE_VIDEO:
            # Match BEV height to camera height for horizontal stack
            h_cam, w_cam = cam.shape[:2]
            h_bev, w_bev = bev_rgb.shape[:2]
            bev_resized = cv2.resize(bev_rgb, (int(w_bev * (h_cam / float(h_bev))), h_cam))
            combined = np.hstack((cam, bev_resized))
            # Draw HUD panel on the combined output as well
            put_hud_panel(combined, hud, org=(10, 30), scale=0.7)
            if vw_demo is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw_demo = cv2.VideoWriter(DEMO_VIDEO_PATH, fourcc, fps_in, (combined.shape[1], combined.shape[0]))
            vw_demo.write(combined)
        timing.end("viz")

        key=cv2.waitKey(1)&0xFF
        if key==27: break
        frame_id+=1

    cap.release(); cv2.destroyAllWindows()
    if vw_demo is not None:
        vw_demo.release()
    print("\n=== TIMING SUMMARY ===")
    print(timing.summary_str())
    if times:
        avg=sum(times)/len(times)
        print(f"Avg FPS: {1.0/avg:0.1f} over {len(times)} frames")

if __name__=="__main__":
    main()
