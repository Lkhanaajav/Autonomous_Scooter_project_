# camera_waypoint_optimized.py
# Ultra-fast real-time road & sidewalk detection and path planning
# Optimizations: Model caching, async processing, simplified algorithms, frame skipping

import os, cv2, math, numpy as np, networkx as nx, sys, torch, time, threading
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from collections import deque
import queue

# =============================================================================
# Optimized Config + Model Loader
# =============================================================================
class OptimizedConfig:
    def __init__(
        self,
        model_dir="models/my-segformer-road_new",
        conf_thresh=0.5,
        road_id=1,
        device=None,
        input_size=(256, 144),  # 👈 Even smaller for speed
        use_fp16=True,
        enable_tensorrt=False,  # For future TensorRT optimization
    ):
        self.model_dir = model_dir
        self.conf_thresh = conf_thresh
        self.road_id = road_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model once and cache
        self.processor = SegformerImageProcessor.from_pretrained(model_dir)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
        
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()
        
        self.input_size = input_size
        self.use_fp16 = use_fp16

class OptimizedFastRoadDetector:
    def __init__(self, cfg: OptimizedConfig):
        self.device = cfg.device
        self.processor = cfg.processor
        self.model = cfg.model.to(self.device)
        self.model.eval()
        self.conf_thresh = cfg.conf_thresh
        self.road_id = cfg.road_id
        self.input_size = cfg.input_size
        self.use_fp16 = cfg.use_fp16
        
        # Pre-allocate tensors for speed
        self._setup_optimizations()
        
    def _setup_optimizations(self):
        """Setup optimizations for faster inference"""
        # Enable optimizations
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        # Pre-compile model for faster inference
        dummy_input = torch.randn(1, 3, self.input_size[1], self.input_size[0])
        if self.use_fp16:
            dummy_input = dummy_input.half()
        dummy_input = dummy_input.to(self.device)
        
        # Warm up the model
        with torch.no_grad():
            _ = self.model(dummy_input)

    def process_frame_fast(self, frame):
        """Ultra-fast frame processing with optimizations"""
        # Resize to smaller input for speed
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor directly (faster than PIL)
        rgb_tensor = torch.from_numpy(rgb_small).permute(2, 0, 1).float() / 255.0
        if self.use_fp16:
            rgb_tensor = rgb_tensor.half()
        
        # Batch dimension
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        
        # Fast inference
        with torch.no_grad():
            outputs = self.model(pixel_values=rgb_tensor)
        
        # Fast upsampling
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(frame.shape[0], frame.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        
        # Get segmentation
        seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return seg

# =============================================================================
# Constants (same as original)
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
    [100, 480],
    [500, 480],
    [400, 100],
    [200, 100]
], dtype=np.float32)

bev_size = (600, 500)
H = cv2.getPerspectiveTransform(src_points, dst_points)
Hinv = np.linalg.inv(H)
TRIM_BOTTOM = 20

PATH_COLORS = [
    (0,255,255), (255,255,0), (255,0,255),
    (0,165,255), (0,255,128), (128,0,255),
    (255,128,0), (0,128,255), (128,255,0)
]

DEFAULT_CAMERA_ID = '/dev/video21'
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
DISPLAY_HEIGHT = 300

# =============================================================================
# Optimized Utilities
# =============================================================================
def split_masks_from_output(model_output, road_id=ROAD_ID, sidewalk_id=SIDEWALK_ID):
    m = model_output.astype(np.uint8)
    uniq = set(np.unique(m).tolist())
    if uniq.issubset({0, 255}):
        sidewalk = (m > 0).astype(np.uint8) * 255
        road = np.zeros_like(sidewalk, dtype=np.uint8)
        return sidewalk, road
    sidewalk = (m == sidewalk_id).astype(np.uint8) * 255
    road = (m == road_id).astype(np.uint8) * 255
    return sidewalk, road

def colorize_sidewalk_road(frame_bgr, sidewalk_mask_255, road_mask_255, alpha=0.45):
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    color_layer[road_mask_255 > 0] = (255, 120, 0)
    color_layer[sidewalk_mask_255 > 0] = (0, 200, 0)
    cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def draw_ribbon(img, pts, color, width=14, glow=6):
    h, w = img.shape[:2]
    pts = [(x, y) for (x, y) in pts if 0 <= x < w and 0 <= y < h]
    if len(pts) < 2:
        return img
    layer = np.zeros_like(img)
    cv2.polylines(layer, [np.int32(pts)], False, color, width, cv2.LINE_AA)
    if glow > 0:
        blur = cv2.GaussianBlur(layer, (0,0), glow)
        img = cv2.addWeighted(img, 1.0, blur, 0.25, 0)
    return cv2.add(img, layer)

def fast_thinning_optimized(binary_0_255):
    """Optimized thinning using OpenCV's built-in function"""
    try:
        import cv2.ximgproc as xip
        return xip.thinning(binary_0_255, thinningType=xip.THINNING_ZHANGSUEN)
    except:
        # Fallback to simple erosion-based thinning
        kernel = np.ones((3,3), np.uint8)
        skeleton = cv2.erode(binary_0_255, kernel, iterations=1)
        return skeleton

def extract_skeleton_graph_fast(bev_binary_0_255, trim_px=5):
    """Fast skeleton extraction with optimizations"""
    # Fast morphological operations
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
    bev_clean = cv2.morphologyEx(bev_binary_0_255, cv2.MORPH_CLOSE, kernel)
    bev_clean = cv2.medianBlur(bev_clean, 3)  # Smaller blur
    _, binary = cv2.threshold(bev_clean, 127, 255, cv2.THRESH_BINARY)
    
    # Fast thinning
    skeleton = fast_thinning_optimized(binary)
    
    if trim_px > 0:
        skeleton[:trim_px, :] = 0
        skeleton[:, :trim_px] = 0
        skeleton[:, -trim_px:] = 0
        skeleton[-trim_px:, :] = 0
    
    # Build graph more efficiently
    G = nx.Graph()
    ys, xs = np.where(skeleton > 0)
    
    # Vectorized neighbor checking
    for (y, x) in zip(ys, xs):
        # Check 8-connected neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                ny, nx_ = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and 
                    0 <= nx_ < skeleton.shape[1] and 
                    skeleton[ny, nx_] == 255):
                    G.add_edge((x, y), (nx_, ny), weight=math.hypot(dx, dy))
    
    return skeleton, G

def skeleton_endpoints(G):
    return [n for n in G.nodes if G.degree[n] == 1]

def project_points_bev_to_cam(points_bev):
    if not points_bev: return []
    pts = np.array(points_bev, np.float32).reshape(-1,1,2)
    cam = cv2.perspectiveTransform(pts, Hinv).reshape(-1,2)
    return [(float(x),float(y)) for x,y in cam]

# =============================================================================
# Caching and State Management
# =============================================================================
class PathCache:
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size
        self.frame_buffer = deque(maxlen=3)  # Keep last 3 frames for temporal consistency
        
    def get_cached_paths(self, frame_hash):
        return self.cache.get(frame_hash, None)
    
    def cache_paths(self, frame_hash, paths, skeleton, graph):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[frame_hash] = (paths, skeleton, graph)
    
    def add_frame(self, frame):
        self.frame_buffer.append(frame)
    
    def get_temporal_consistency(self):
        """Use temporal information for smoother paths"""
        if len(self.frame_buffer) < 2:
            return None
        return self.frame_buffer[-1]  # Use previous frame info

# =============================================================================
# Async Processing Pipeline
# =============================================================================
class AsyncProcessor:
    def __init__(self, model, cache):
        self.model = model
        self.cache = cache
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_worker)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _process_worker(self):
        """Background processing worker"""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                    
                frame, frame_id = frame_data
                
                # Process frame
                model_out = self.model.process_frame_fast(frame)
                sidewalk_mask, road_mask = split_masks_from_output(model_out)
                
                # BEV transforms
                bev_sidewalk = cv2.warpPerspective(sidewalk_mask, H, bev_size)
                bev_road = cv2.warpPerspective(road_mask, H, bev_size)
                if TRIM_BOTTOM > 0:
                    bev_sidewalk = bev_sidewalk[:-TRIM_BOTTOM, :]
                    bev_road = bev_road[:-TRIM_BOTTOM, :]
                
                # Fast skeleton + graph
                skeleton_mask, graph = extract_skeleton_graph_fast(bev_sidewalk, trim_px=5)
                endpoints = skeleton_endpoints(graph)
                start = max(endpoints, key=lambda p: p[1]) if endpoints else None
                
                # Put result in queue
                result = {
                    'frame_id': frame_id,
                    'model_out': model_out,
                    'sidewalk_mask': sidewalk_mask,
                    'road_mask': road_mask,
                    'bev_sidewalk': bev_sidewalk,
                    'bev_road': bev_road,
                    'skeleton_mask': skeleton_mask,
                    'graph': graph,
                    'endpoints': endpoints,
                    'start': start
                }
                
                self.result_queue.put(result, timeout=0.1)
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def add_frame(self, frame, frame_id):
        try:
            self.frame_queue.put((frame, frame_id), timeout=0.01)
        except queue.Full:
            pass  # Skip frame if queue is full
    
    def get_result(self):
        try:
            return self.result_queue.get(timeout=0.01)
        except queue.Empty:
            return None

# =============================================================================
# Main Optimized Loop
# =============================================================================
def initialize_optimized_model():
    cfg = OptimizedConfig(model_dir="models/my-segformer-road_new", conf_thresh=0.5, road_id=ROAD_ID)
    return OptimizedFastRoadDetector(cfg)

def process_camera_optimized(output_dir, stride=5, save_video=False, camera_id=DEFAULT_CAMERA_ID):
    print("🔧 Initializing Optimized FastRoadDetector...")
    model = initialize_optimized_model()
    print("✅ Model ready!")
    
    # Initialize async processor
    cache = PathCache()
    async_processor = AsyncProcessor(model, cache)
    async_processor.start()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📹 Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera {camera_id}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    vw = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(os.path.join(output_dir,"cam_paths.mp4"),fourcc,CAMERA_FPS,(CAMERA_WIDTH,CAMERA_HEIGHT))
    
    frame_id = 0
    last_result = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera frame lost, retrying...")
                continue
            
            # Add frame to async processor
            if frame_id % stride == 0:
                async_processor.add_frame(frame, frame_id)
            
            # Get latest result
            result = async_processor.get_result()
            if result is not None:
                last_result = result
            
            # Use last available result for display
            if last_result is not None:
                # Create overlays
                cam_overlay = colorize_sidewalk_road(frame, 
                                                   last_result['sidewalk_mask'], 
                                                   last_result['road_mask'])
                
                # BEV visualization
                H_bev, W_bev = last_result['skeleton_mask'].shape
                bev_color = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)
                bev_color[last_result['bev_road'] > 0] = (255, 120, 0)
                bev_color[last_result['bev_sidewalk'] > 0] = (0, 200, 0)
                gy, gx = np.where(last_result['skeleton_mask'] > 0)
                bev_color[gy, gx] = (0, 128, 0)
                
                # Camera overlay with paths
                cam_paths = cam_overlay.copy()
                if last_result['start']:
                    other_endpoints = [e for e in last_result['endpoints'] if e != last_result['start']]
                    for idx, end in enumerate(other_endpoints[:3]):  # Limit to 3 paths for speed
                        try:
                            path = nx.dijkstra_path(last_result['graph'], 
                                                  last_result['start'], end, weight="weight")
                        except nx.NetworkXNoPath:
                            continue
                        color = PATH_COLORS[idx % len(PATH_COLORS)]
                        
                        # Draw on BEV
                        for i in range(len(path)-1):
                            x1, y1 = path[i]; x2, y2 = path[i+1]
                            cv2.line(bev_color, (int(x1),int(y1)), (int(x2),int(y2)), color, 2, cv2.LINE_AA)
                        
                        # Draw on camera
                        cam_pts = project_points_bev_to_cam(path)
                        cam_paths = draw_ribbon(cam_paths, cam_pts, color=color, width=18, glow=6)
                
                # Save if needed
                if save_video and frame_id % stride == 0:
                    cv2.imwrite(os.path.join(output_dir,f"bev_paths_{frame_id:04d}.png"), bev_color)
                    cv2.imwrite(os.path.join(output_dir,f"cam_paths_{frame_id:04d}.png"), cam_paths)
                if vw: vw.write(cam_paths)
                
                # Display
                bev_display = cv2.resize(bev_color, (int(DISPLAY_HEIGHT * bev_color.shape[1] / bev_color.shape[0]), DISPLAY_HEIGHT))
                cam_display = cv2.resize(cam_paths, (int(DISPLAY_HEIGHT * cam_paths.shape[1] / cam_paths.shape[0]), DISPLAY_HEIGHT))
                combined = np.hstack((cam_display, bev_display))
                
                cv2.imshow("Optimized Real-time Path Planning (Camera | BEV)", combined)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("🛑 Stopping camera processing...")
                break
            
            frame_id += 1
            
    finally:
        # Cleanup
        async_processor.stop()
        cap.release()
        if vw: vw.release()
        cv2.destroyAllWindows()
        print("✅ Done. Results in", output_dir)

if __name__ == "__main__":
    camera_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CAMERA_ID
    process_camera_optimized("camera_results_optimized", stride=3, save_video=False, camera_id=camera_id)
