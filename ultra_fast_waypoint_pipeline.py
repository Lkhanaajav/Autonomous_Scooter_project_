# waypoint_skeleton_pipeline_v4.py
# Segmentation → BEV → Distance-based BEV cleaning (removes thin branches)
# → Zhang–Suen Skeleton → Pruned Main Path → Evenly Spaced Waypoints

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
SCALE_SKELETON = 0.5         # compute skeleton at this scale, then upsample
PRUNE_BRANCH_LEN = 18        # pixels to prune tiny skeleton branches
WAYPOINT_ARCSTEP = 40.0      # BEV pixels between waypoints (arc-length sampling)
WAYPOINT_COUNT_MAX = 12
EMA_ALPHA = 0.35             # temporal smoothing for waypoints (0..1)
DT_CORE_THRESH = 8.0         # distance (in px) kept from BEV edges (raise to trim more branches)
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

# ========= BEV cleaning (distance-based, removes thin branches) =========
def clean_sidewalk_mask(bev_mask_255, dt_thresh=DT_CORE_THRESH):
    """
    Remove narrow peninsulas / side bumps before skeletonization using
    a distance-transform 'core' threshold.
    """
    mask = bev_mask_255.copy().astype(np.uint8)

    # Basic smooth/close to stabilize DT
    k7 = np.ones((7,7), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)

    # Distance to boundary; keep only the interior core
    dist = cv2.distanceTransform((mask>0).astype(np.uint8), cv2.DIST_L2, 5)
    core = (dist > float(dt_thresh)).astype(np.uint8) * 255

    # Soft restore of boundary thickness so skeleton stays centered but continuous
    core = cv2.dilate(core, k5, iterations=1)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k5, iterations=1)
    return core

# ========= Fast Zhang–Suen =========
def zhang_suen_thinning(img0):
    img = (img0>0).astype(np.uint8)
    prev = np.zeros_like(img)
    while True:
        P2 = np.roll(img, -1, 0); P3 = np.roll(np.roll(img,-1,0),-1,1); P4 = np.roll(img, -1, 1)
        P5 = np.roll(np.roll(img,  1,0),-1,1); P6 = np.roll(img, 1, 0);   P7 = np.roll(np.roll(img,1,0),1,1)
        P8 = np.roll(img, 1, 1);  P9 = np.roll(np.roll(img,-1,0), 1,1)
        N = P2+P3+P4+P5+P6+P7+P8+P9
        C = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+\
            ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+\
            ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+\
            ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m1=(img==1)&(N>=2)&(N<=6)&(C==1)&((P2*P4*P6)==0)&((P4*P6*P8)==0)
        img[m1]=0
        P2 = np.roll(img, -1, 0); P3 = np.roll(np.roll(img,-1,0),-1,1); P4 = np.roll(img, -1, 1)
        P5 = np.roll(np.roll(img,  1,0),-1,1); P6 = np.roll(img, 1, 0);   P7 = np.roll(np.roll(img,1,0),1,1)
        P8 = np.roll(img, 1, 1);  P9 = np.roll(np.roll(img,-1,0), 1,1)
        N = P2+P3+P4+P5+P6+P7+P8+P9
        C = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+\
            ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+\
            ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+\
            ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m2=(img==1)&(N>=2)&(N<=6)&(C==1)&((P2*P4*P8)==0)&((P2*P6*P8)==0)
        img[m2]=0
        if not np.any(img!=prev): break
        prev[:] = img
    return (img*255).astype(np.uint8)

# ========= Skeleton + Waypoint utilities =========
def prune_small_branches(skel, min_len=PRUNE_BRANCH_LEN):
    s = skel.copy()
    for _ in range(min_len):
        nb = cv2.filter2D((s>0).astype(np.uint8), -1, np.ones((3,3), np.uint8))
        endpoints = ((s>0) & (nb==2))  # 1 neighbor + self
        s[endpoints] = 0
    return s

def main_path_from_bottom_center(skel):
    """
    Keep only the skeleton component connected to the bottom-center area.
    Removes all stray paths and side branches.
    """
    h, w = skel.shape
    bin_ = (skel > 0).astype(np.uint8)

    # Connected components
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_)
    if num <= 1:
        return skel

    # Define bottom center window (start zone)
    x_center = w // 2
    y_bottom = h - 1
    zone_w = int(w * 0.25)     # 25% of width
    zone_h = int(h * 0.15)     # 15% of height near bottom
    x1, x2 = x_center - zone_w//2, x_center + zone_w//2
    y1, y2 = h - zone_h, h

    # Determine which component touches bottom region
    main_label = None
    best_area = 0
    for cid in range(1, num):
        comp_mask = (labels == cid)
        # Check intersection with bottom zone
        if np.any(comp_mask[y1:y2, x1:x2]):
            area = stats[cid, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                main_label = cid

    if main_label is None:
        # fallback to largest component
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

# ========= Extract Skeleton =========
def extract_skeleton(bev_binary_0_255, trim_px=5):
    # Binary, light denoise
    kernel = np.ones((5,5),np.uint8)
    bev_clean = cv2.morphologyEx(bev_binary_0_255, cv2.MORPH_CLOSE, kernel)
    bev_clean = cv2.medianBlur(bev_clean,5)
    _,binary=cv2.threshold(bev_clean,127,255,cv2.THRESH_BINARY)

    # scale down for speed
    h0,w0 = binary.shape
    small = cv2.resize(binary,(max(1,int(w0*SCALE_SKELETON)),max(1,int(h0*SCALE_SKELETON))),interpolation=cv2.INTER_NEAREST)
    skeleton_small = zhang_suen_thinning(small)
    skeleton = cv2.resize(skeleton_small,(w0,h0),interpolation=cv2.INTER_NEAREST)

    if trim_px>0:
        skeleton[:trim_px,:]=0; skeleton[-trim_px:,:]=0
        skeleton[:, :trim_px]=0; skeleton[:, -trim_px:]=0
    return skeleton

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
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    Hc=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

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

        # 3) BEV + distance-core cleaning
        timing.start("bev")
        bev_sidewalk=cv2.warpPerspective(sidewalk_mask,H,BEV_SIZE)
        if TRIM_BOTTOM>0: bev_sidewalk=bev_sidewalk[:-TRIM_BOTTOM,:]
        bev_sidewalk = clean_sidewalk_mask(bev_sidewalk, DT_CORE_THRESH)
        timing.end("bev")

        # 4) Skeleton
        timing.start("skeleton")
        skel = extract_skeleton(bev_sidewalk, trim_px=5)
        skel = prune_small_branches(skel, PRUNE_BRANCH_LEN)
        h_b, w_b = skel.shape
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

        # 6) Visualization
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
             f"Infer {timing.timings['inference'][-1]*1000 if timing.timings['inference'] else 0:.1f} ms",
             f"Skeleton {timing.timings['skeleton'][-1]*1000 if timing.timings['skeleton'] else 0:.1f} ms",
             f"Waypoints {len(waypoints_cam)}"]
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
