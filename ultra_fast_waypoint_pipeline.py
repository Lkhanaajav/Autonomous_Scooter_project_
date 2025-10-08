# -*- coding: utf-8 -*-
# waypoint_skeleton_pipeline_v5.py
# Segmentation → BEV → Distance-based BEV cleaning (removes thin branches)
# → Guo–Hall Skeleton → Pruned Main Path → Waypoints → Direction Commands

import os, cv2, time, math
import numpy as np
from collections import defaultdict
from fast_road_detector import FastRoadDetector, Config

# ========= Settings =========
ROAD_ID = 1
SIDEWALK_ID = 2
MODEL_DIR = "models/my-segformer-road"
CONF_THRESH = 0.5
INPUT = "test_video_june_03_3.mp4"   # or int(0) for webcam
STRIDE = 4
SHOW_SEG = True
SAVE_VIDEO = False
BEV_SIZE = (600, 500)        # (W, H)
TRIM_BOTTOM = 30
CAM_FPS_HINT = 30

# Tuning knobs
SCALE_SKELETON = 0.5
PRUNE_BRANCH_LEN = 18
WAYPOINT_ARCSTEP = 40.0
WAYPOINT_COUNT_MAX = 12
EMA_ALPHA = 0.35
DT_CORE_THRESH = 8.0
TURN_THRESH_DEG = 20.0       # angle threshold for left/right turns
# ============================

# ---- Homography ----
src_points = np.array([
    [0.0,   717.0],
    [1278.0, 717.0],
    [860.0,  337.0],
    [573.0,  329.0]
], dtype=np.float32)
dst_points = np.array([[100,480],[500,480],[400,100],[200,100]],dtype=np.float32)
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
    zone_w = int(w * 0.25)
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
def compute_commands(waypoints, turn_thresh_deg=TURN_THRESH_DEG):
    """
    Given ordered waypoints [(x,y), ...], compute simple textual commands.
    Returns list like ['STRAIGHT', 'LEFT', 'RIGHT', 'STRAIGHT', ...]
    """
    if len(waypoints) < 3:
        return []
    cmds = []
    def heading(p1, p2):
        return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
    for i in range(1, len(waypoints)-1):
        h1 = heading(waypoints[i-1], waypoints[i])
        h2 = heading(waypoints[i], waypoints[i+1])
        dtheta = (h2 - h1 + 180) % 360 - 180
        if abs(dtheta) < turn_thresh_deg:
            cmds.append("STRAIGHT")
        elif dtheta > 0:
            cmds.append("TURN LEFT")
        else:
            cmds.append("TURN RIGHT")
    cmds.append("STOP")
    return cmds

# ========= HUD =========
def put_hud(img, lines, org=(10,25), scale=0.6, color=(255,255,255)):
    y=org[1]
    for text in lines:
        cv2.putText(img,text,(org[0],y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)
        y+=int(24*scale)

# ========= Main =========
def main():
    cfg=Config(model_dir=MODEL_DIR, conf_thresh=CONF_THRESH, road_id=ROAD_ID)
    model=FastRoadDetector(cfg)
    cap=cv2.VideoCapture(INPUT if not str(INPUT).isdigit() else int(INPUT))
    if not cap.isOpened():
        print(f"ERROR: cannot open {INPUT}"); return
    fps_in=cap.get(cv2.CAP_PROP_FPS) or CAM_FPS_HINT
    frame_id=0; times=[]; last_mask=None; last_waypoints_cam=None

    print("Running waypoint pipeline… (ESC to quit)")

    while True:
        t0=time.time()
        ok,frame=cap.read()
        if not ok: break
        run_net=(frame_id%STRIDE==0)

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
        if TRIM_BOTTOM>0: bev_sidewalk=bev_sidewalk[:-TRIM_BOTTOM,:]
        bev_sidewalk = clean_sidewalk_mask(bev_sidewalk, DT_CORE_THRESH)
        timing.end("bev")

        # 4) Skeleton
        timing.start("skeleton")
        skel = extract_skeleton(bev_sidewalk, trim_px=5)
        skel = prune_small_branches(skel, PRUNE_BRANCH_LEN)
        main_skel = main_path_from_bottom_center(skel)
        timing.end("skeleton")

        # 5) Waypoints
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

        # 6) Commands
        commands = compute_commands(waypoints_bev)
        if commands:
            print(" → ".join(commands))
            current_cmd = commands[0]
        else:
            current_cmd = "..."
        
        # 7) Visualization
        timing.start("viz")
        bev_rgb=np.zeros((bev_sidewalk.shape[0],bev_sidewalk.shape[1],3),np.uint8)
        bev_rgb[bev_sidewalk>0]=(0,80,0)
        bev_rgb[skel>0]=(255,255,0)
        bev_rgb[main_skel>0]=(0,255,255)

        cam=frame.copy()
        if SHOW_SEG:
            overlay=np.zeros_like(cam)
            overlay[road_mask>0]=(0,140,255)
            overlay[sidewalk_mask>0]=(0,200,0)
            cam=cv2.addWeighted(overlay,0.35,cam,1.0,0)
        for (x,y) in waypoints_cam:
            cv2.circle(cam,(x,y),6,(0,255,255),-1)
        if len(waypoints_cam)>=2:
            cv2.line(cam, waypoints_cam[0], waypoints_cam[min(3,len(waypoints_cam)-1)], (0,255,255), 2, cv2.LINE_AA)

        dt=time.time()-t0
        times.append(dt)
        fps=1.0/dt if dt>0 else 0
        hud=[f"Frame {frame_id}",f"FPS {fps:4.1f}",
             f"Skeleton {timing.timings['skeleton'][-1]*1000 if timing.timings['skeleton'] else 0:.1f} ms",
             f"Waypoints {len(waypoints_cam)}",
             f"Command: {current_cmd}"]
        put_hud(cam,hud,org=(10,28),scale=0.7)

        cv2.imshow("Cam",cam)
        cv2.imshow("BEV",bev_rgb)
        timing.end("viz")

        key=cv2.waitKey(1)&0xFF
        if key==27: break
        frame_id+=1

    cap.release(); cv2.destroyAllWindows()
    print("\n=== TIMING SUMMARY ===")
    print(timing.summary_str())
    if times:
        avg=sum(times)/len(times)
        print(f"Avg FPS: {1.0/avg:0.1f} over {len(times)} frames")

if __name__=="__main__":
    main()
