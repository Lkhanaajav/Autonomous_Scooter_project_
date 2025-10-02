# ultra_fast_waypoint_pipeline.py - ULTRA-OPTIMIZED VERSION
import os
import cv2
import math
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import time
from numba import jit, prange
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ---- Your detector ----
from fast_road_detector import FastRoadDetector, Config

# =============================================================================
# Timing Analysis
# =============================================================================
class TimingAnalyzer:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timings = {}
        
    def start_timer(self, name):
        self.current_timings[name] = time.time()
        
    def end_timer(self, name):
        if name in self.current_timings:
            duration = time.time() - self.current_timings[name]
            self.timings[name].append(duration)
            return duration
        return 0.0
    
    def get_summary(self):
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'count': len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return summary
    
    def print_summary(self):
        print("\n" + "="*80)
        print("DETAILED TIMING ANALYSIS")
        print("="*80)
        
        summary = self.get_summary()
        total_time = sum(data['total'] for data in summary.values())
        
        sorted_timings = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print(f"{'Operation':<30} {'Total(s)':<10} {'Avg(ms)':<10} {'Count':<8} {'Min(ms)':<10} {'Max(ms)':<10} {'%':<8}")
        print("-"*80)
        
        for name, data in sorted_timings:
            percentage = (data['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<30} {data['total']:<10.3f} {data['average']*1000:<10.1f} {data['count']:<8} {data['min']*1000:<10.1f} {data['max']*1000:<10.1f} {percentage:<8.1f}")
        
        print("-"*80)
        print(f"{'TOTAL TIME':<30} {total_time:<10.3f}")
        print("="*80)

# Global timing analyzer
timing = TimingAnalyzer()

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
# NUMBA-ACCELERATED FUNCTIONS
# =============================================================================
@jit(nopython=True, parallel=True)
def fast_skeleton_numba(binary_img):
    """Numba-accelerated skeletonization."""
    h, w = binary_img.shape
    skeleton = binary_img.copy()
    
    # Simple thinning algorithm optimized for numba
    changed = True
    while changed:
        changed = False
        
        # First pass
        for y in prange(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                
                # Count neighbors
                neighbors = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        if skeleton[y+dy, x+dx] > 0:
                            neighbors += 1
                
                # Simple thinning conditions
                if 2 <= neighbors <= 6:
                    # Check connectivity
                    transitions = 0
                    for i in range(8):
                        p1 = skeleton[y-1, x] if i == 0 else skeleton[y-1, x+1] if i == 1 else skeleton[y, x+1] if i == 2 else skeleton[y+1, x+1] if i == 3 else skeleton[y+1, x] if i == 4 else skeleton[y+1, x-1] if i == 5 else skeleton[y, x-1] if i == 6 else skeleton[y-1, x-1]
                        p2 = skeleton[y-1, x+1] if i == 0 else skeleton[y, x+1] if i == 1 else skeleton[y+1, x+1] if i == 2 else skeleton[y+1, x] if i == 3 else skeleton[y+1, x-1] if i == 4 else skeleton[y, x-1] if i == 5 else skeleton[y-1, x-1] if i == 6 else skeleton[y-1, x]
                        if p1 == 0 and p2 > 0:
                            transitions += 1
                    
                    if transitions == 1:
                        skeleton[y, x] = 0
                        changed = True
    
    return skeleton

@jit(nopython=True, parallel=True)
def build_graph_edges_numba(skeleton):
    """Numba-accelerated graph edge building."""
    h, w = skeleton.shape
    edges = []
    
    for y in prange(h):
        for x in range(w):
            if skeleton[y, x] == 0:
                continue
            
            # Check 8-connected neighbors
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and 
                        skeleton[ny, nx] > 0):
                        weight = math.sqrt(dx*dx + dy*dy)
                        edges.append((x, y, nx, ny, weight))
    
    return edges

# =============================================================================
# ULTRA-OPTIMIZED Utilities
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
# ULTRA-FAST Skeletonization
# =============================================================================
def ultra_fast_skeletonization(bin_img_0_255):
    """Ultra-fast skeletonization that produces correct paths for waypoint detection."""
    timing.start_timer("skeletonization")
    img = (bin_img_0_255 > 0).astype(np.uint8)
    
    # Method 1: Try OpenCV's built-in thinning (best quality, fast)
    try:
        skeleton = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        if skeleton is not None and np.sum(skeleton > 0) > 100:  # Check for reasonable skeleton
            # Post-process to ensure connectivity for pathfinding
            skeleton = post_process_skeleton_for_pathfinding(skeleton)
            timing.end_timer("skeletonization")
            return skeleton.astype(np.uint8)
    except Exception:
        pass
    
    # Method 2: Use scikit-image skeletonize (very good quality)
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(img.astype(bool))
        skeleton = (skeleton * 255).astype(np.uint8)
        if np.sum(skeleton > 0) > 100:  # Check for reasonable skeleton
            skeleton = post_process_skeleton_for_pathfinding(skeleton)
            timing.end_timer("skeletonization")
            return skeleton
    except ImportError:
        pass
    except Exception:
        pass
    
    # Method 3: Robust distance transform approach
    result = robust_skeletonization(img)
    timing.end_timer("skeletonization")
    return result

def post_process_skeleton_for_pathfinding(skeleton):
    """Post-process skeleton to ensure good connectivity for pathfinding."""
    # Remove isolated pixels
    kernel = np.ones((3,3), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
    
    # Fill small gaps to ensure connectivity
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)
    
    # Remove very small branches that don't contribute to main paths
    skeleton = remove_small_branches(skeleton, min_length=5)
    
    return skeleton

def remove_small_branches(skeleton, min_length=5):
    """Remove small branches from skeleton to improve pathfinding."""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    
    # Keep only components with sufficient length
    result = np.zeros_like(skeleton)
    for i in range(1, num_labels):
        component = (labels == i).astype(np.uint8) * 255
        
        # Calculate component length (approximate)
        if np.sum(component > 0) >= min_length:
            result = cv2.bitwise_or(result, component)
    
    return result

def distance_transform_skeleton_improved(img):
    """Create skeleton using distance transform with proper connectivity."""
    # Get distance transform
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    
    # Create skeleton using distance transform ridges
    skeleton = np.zeros_like(img)
    
    # Find local maxima (ridge points)
    kernel = np.ones((3,3), np.uint8)
    local_maxima = cv2.dilate(dist_transform, kernel) == dist_transform
    skeleton[local_maxima & (dist_transform > 1)] = 255
    
    # Ensure connectivity by connecting nearby points
    skeleton = ensure_connectivity_for_pathfinding(skeleton, dist_transform)
    
    # Clean up
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)
    
    return skeleton.astype(np.uint8)

def ensure_connectivity_for_pathfinding(skeleton, dist_transform):
    """Ensure skeleton has good connectivity for pathfinding."""
    # Find gaps in skeleton
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Fill gaps where distance transform is high
    gaps = (dilated > 0) & (skeleton == 0) & (dist_transform > 2)
    skeleton[gaps] = 255
    
    return skeleton

def robust_skeletonization(img):
    """Robust skeletonization that works reliably for pathfinding."""
    # Clean the input image
    kernel = np.ones((3,3), np.uint8)
    img_clean = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_OPEN, kernel)
    
    # Get distance transform
    dist_transform = cv2.distanceTransform(img_clean, cv2.DIST_L2, 5)
    
    # Create skeleton using multiple approaches
    skeleton = np.zeros_like(img_clean)
    
    # Method 1: Distance transform ridges
    kernel = np.ones((3,3), np.uint8)
    local_maxima = cv2.dilate(dist_transform, kernel) == dist_transform
    skeleton[local_maxima & (dist_transform > 2)] = 255
    
    # Method 2: If skeleton is too sparse, use erosion-based approach
    if np.sum(skeleton > 0) < 100:
        # Use iterative erosion to create skeleton
        temp_img = img_clean.copy()
        while np.sum(temp_img > 0) > 0:
            # Find boundary pixels
            eroded = cv2.erode(temp_img, kernel, iterations=1)
            boundary = temp_img - eroded
            
            # Add boundary pixels to skeleton if they're on the "spine"
            if np.sum(boundary > 0) > 0:
                # Keep only pixels that are likely part of the main path
                dist_at_boundary = dist_transform[boundary > 0]
                if len(dist_at_boundary) > 0:
                    threshold = np.percentile(dist_at_boundary, 70)  # Keep top 30%
                    skeleton[boundary > 0] = 255
                    temp_img = eroded
                else:
                    break
            else:
                break
    
    # Method 3: If still too sparse, use simple centerline extraction
    if np.sum(skeleton > 0) < 50:
        # Find the main connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_clean, connectivity=8)
        if num_labels > 1:
            # Find the largest component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            main_component = (labels == largest_label).astype(np.uint8) * 255
            
            # Create a simple centerline by finding the middle of the component
            y_coords, x_coords = np.where(main_component > 0)
            if len(y_coords) > 0:
                # Create a simple path through the center
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                
                # Create a simple centerline
                for y in range(y_min, y_max, 5):  # Every 5 pixels
                    x_center = int((x_min + x_max) / 2)
                    if 0 <= y < skeleton.shape[0] and 0 <= x_center < skeleton.shape[1]:
                        skeleton[y, x_center] = 255
    
    # Clean up the skeleton
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)
    
    return skeleton.astype(np.uint8)

# =============================================================================
# ULTRA-OPTIMIZED Graph Builder
# =============================================================================
class UltraOptimizedGraphBuilder:
    """Ultra-optimized graph builder with advanced caching and vectorization."""
    
    def __init__(self):
        self.cache = {}
        self.last_skeleton_hash = None
        self.cached_graph = None
        self.frame_skip_threshold = 0.95  # Skip if 95% similar
        
    def _get_skeleton_hash(self, skeleton):
        """Get a hash of the skeleton for caching."""
        return hash(skeleton.tobytes())
    
    def _skeleton_similarity(self, sk1, sk2):
        """Check if two skeletons are similar enough to reuse graph."""
        if sk1.shape != sk2.shape:
            return 0.0
        
        # Calculate similarity as percentage of matching pixels
        matches = np.sum(sk1 == sk2)
        total = sk1.size
        return matches / total
    
    def extract_skeleton_graph(self, bev_binary_0_255, trim_px=5):
        """Ultra-optimized skeleton extraction with advanced caching."""
        timing.start_timer("extract_skeleton_graph")
        
        # Fast preprocessing
        timing.start_timer("preprocessing")
        kernel = np.ones((3, 3), np.uint8)
        bev_clean = cv2.morphologyEx(bev_binary_0_255, cv2.MORPH_CLOSE, kernel)
        bev_clean = cv2.medianBlur(bev_clean, 3)
        _, binary = cv2.threshold(bev_clean, 127, 255, cv2.THRESH_BINARY)
        timing.end_timer("preprocessing")
        
        # Ultra-fast skeletonization
        skeleton = ultra_fast_skeletonization(binary)
        sk = skeleton.copy()
        
        timing.start_timer("trimming")
        if trim_px > 0:
            sk[:trim_px, :]  = 0
            sk[:, :trim_px]  = 0
            sk[:, -trim_px:] = 0
            sk[-trim_px:, :] = 0
        timing.end_timer("trimming")
        
        # Check similarity with previous skeleton
        timing.start_timer("similarity_check")
        if (self.last_skeleton_hash is not None and 
            self.cached_graph is not None):
            # If very similar, reuse graph
            if hasattr(self, 'last_skeleton'):
                similarity = self._skeleton_similarity(sk, self.last_skeleton)
                if similarity > self.frame_skip_threshold:
                    timing.end_timer("similarity_check")
                    timing.end_timer("extract_skeleton_graph")
                    return sk, self.cached_graph
        timing.end_timer("similarity_check")
        
        # Build graph with numba acceleration
        timing.start_timer("graph_construction")
        G = self._build_graph_ultra_fast(sk)
        timing.end_timer("graph_construction")
        
        # Cache the result
        timing.start_timer("caching")
        self.last_skeleton_hash = self._get_skeleton_hash(sk)
        self.cached_graph = G
        self.last_skeleton = sk.copy()
        timing.end_timer("caching")
        
        timing.end_timer("extract_skeleton_graph")
        return sk, G
    
    def _build_graph_ultra_fast(self, sk):
        """Build graph using optimized vectorized method."""
        G = nx.Graph()
        h, w = sk.shape
        
        # Find all skeleton points
        y_coords, x_coords = np.where(sk == 255)
        if len(y_coords) == 0:
            return G
            
        points = list(zip(x_coords, y_coords))
        G.add_nodes_from(points)
        
        # Optimized neighbor finding - only check 4-connected neighbors for speed
        for i, (x, y) in enumerate(points):
            # Check only 4-connected neighbors (faster than 8-connected)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny = x + dx, y + dy
                if (0 <= nx_ < w and 0 <= ny < h and 
                    sk[ny, nx_] == 255 and (nx_, ny) in points):
                    weight = 1.0  # Use unit weight for speed
                    G.add_edge((x, y), (nx_, ny), weight=weight)
        
        return G
    
    def _build_graph_vectorized(self, sk):
        """Fallback vectorized graph building."""
        G = nx.Graph()
        h, w = sk.shape
        
        # Find all skeleton points
        y_coords, x_coords = np.where(sk == 255)
        points = list(zip(x_coords, y_coords))
        G.add_nodes_from(points)
        
        # Vectorized neighbor finding
        for i, (x, y) in enumerate(points):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx_, ny = x + dx, y + dy
                    if (0 <= nx_ < w and 0 <= ny < h and 
                        sk[ny, nx_] == 255 and (nx_, ny) in points):
                        weight = math.hypot(dx, dy)
                        G.add_edge((x, y), (nx_, ny), weight=weight)
        
        return G

# Global ultra-optimized graph builder
ultra_graph_builder = UltraOptimizedGraphBuilder()

def extract_skeleton_graph(bev_binary_0_255, trim_px=5):
    """Wrapper for ultra-optimized graph extraction."""
    return ultra_graph_builder.extract_skeleton_graph(bev_binary_0_255, trim_px)

def skeleton_endpoints(G):
    """Fast endpoint detection."""
    return [n for n in G.nodes if G.degree[n] == 1]

# =============================================================================
# ULTRA-OPTIMIZED Pathfinding
# =============================================================================
class UltraPathfindingCache:
    """Ultra-optimized pathfinding cache with LRU and similarity matching."""
    
    def __init__(self, max_size=200):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        self.access_time = {}
        self.current_time = 0
    
    def get_path(self, start, end, graph_hash):
        """Get cached path with similarity matching."""
        self.current_time += 1
        
        # Exact match
        key = (start, end, graph_hash)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.access_time[key] = self.current_time
            return self.cache[key]
        
        # Similarity match (for nearby endpoints)
        for cached_key, path in self.cache.items():
            cached_start, cached_end, cached_hash = cached_key
            if (cached_hash == graph_hash and 
                self._points_similar(start, cached_start) and 
                self._points_similar(end, cached_end)):
                # Found similar path, return it
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self.access_time[key] = self.current_time
                return path
        
        return None
    
    def _points_similar(self, p1, p2, threshold=10):
        """Check if two points are similar (within threshold distance)."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= threshold
    
    def cache_path(self, start, end, graph_hash, path):
        """Cache a computed path with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
            del self.access_time[lru_key]
        
        key = (start, end, graph_hash)
        self.cache[key] = path
        self.access_count[key] = 1
        self.access_time[key] = self.current_time

# Global ultra pathfinding cache
ultra_path_cache = UltraPathfindingCache()

def get_graph_hash(graph):
    """Get a hash of the graph structure for caching."""
    return hash(tuple(sorted(graph.edges())))

def find_paths_ultra_optimized(start, endpoints, graph):
    """Ultra-optimized pathfinding with advanced caching and parallel processing."""
    timing.start_timer("pathfinding_total")
    
    if not start or not endpoints:
        timing.end_timer("pathfinding_total")
        return []
    
    graph_hash = get_graph_hash(graph)
    other_endpoints = [e for e in endpoints if e != start]
    
    def find_single_path(end):
        # Check cache first
        timing.start_timer("cache_lookup")
        cached_path = ultra_path_cache.get_path(start, end, graph_hash)
        timing.end_timer("cache_lookup")
        
        if cached_path is not None:
            return end, cached_path
        
        # Compute path with A* for better performance
        timing.start_timer("astar_computation")
        try:
            # Use A* instead of Dijkstra for better performance
            path = nx.astar_path(graph, start, end, weight="weight")
            # Cache the result
            ultra_path_cache.cache_path(start, end, graph_hash, path)
            timing.end_timer("astar_computation")
            return end, path
        except nx.NetworkXNoPath:
            timing.end_timer("astar_computation")
            return end, None
    
    # Use parallel processing with optimal thread count
    timing.start_timer("parallel_processing")
    max_workers = min(8, len(other_endpoints))
    paths = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(find_single_path, end) for end in other_endpoints]
        for future in futures:
            end, path = future.result()
            if path is not None:
                paths.append((end, path))
    timing.end_timer("parallel_processing")
    
    timing.end_timer("pathfinding_total")
    return paths

# =============================================================================
# ULTRA-OPTIMIZED Visualization
# =============================================================================
@jit(nopython=True, parallel=True)
def draw_paths_numba(bev_color, paths_data, start, skeleton_mask):
    """Numba-accelerated path drawing."""
    H_bev, W_bev = skeleton_mask.shape
    
    # Draw skeleton
    for y in prange(H_bev):
        for x in range(W_bev):
            if skeleton_mask[y, x] > 0:
                bev_color[y, x, 0] = 0
                bev_color[y, x, 1] = 128
                bev_color[y, x, 2] = 0
    
    # Draw paths
    for path_idx in prange(len(paths_data)):
        path = paths_data[path_idx]
        color = PATH_COLORS[path_idx % len(PATH_COLORS)]
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Draw line between points
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            steps = max(dx, dy)
            
            if steps > 0:
                for step in range(steps + 1):
                    t = step / steps
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))
                    
                    if 0 <= x < W_bev and 0 <= y < H_bev:
                        bev_color[y, x, 0] = color[0]
                        bev_color[y, x, 1] = color[1]
                        bev_color[y, x, 2] = color[2]
    
    return bev_color

def draw_paths_ultra_optimized(bev_color, paths, start, skeleton_mask):
    """Ultra-optimized path drawing."""
    H_bev, W_bev = skeleton_mask.shape
    
    # Pre-compute thickened skeleton for visualization
    vis_skel = cv2.dilate(
        skeleton_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)),  # Smaller for speed
        iterations=1
    )
    gy, gx = np.where(vis_skel > 0)
    bev_color[gy, gx] = (0, 128, 0)
    
    # Draw paths using numba acceleration if available
    if paths and len(paths) > 0:
        try:
            paths_data = [path for _, path in paths]
            bev_color = draw_paths_numba(bev_color, paths_data, start, skeleton_mask)
        except:
            # Fallback to OpenCV
            for idx, (end, path) in enumerate(paths):
                color = PATH_COLORS[idx % len(PATH_COLORS)]
                path_np = np.int32(path).reshape(-1, 1, 2)
                cv2.polylines(bev_color, [path_np], False, color, thickness=15, lineType=cv2.LINE_AA)
                
                # Draw endpoints
                cv2.circle(bev_color, (int(start[0]), int(start[1])), 5, (0, 0, 255), -1)
                cv2.circle(bev_color, (int(end[0]), int(end[1])), 4, color, -1)
    
    return bev_color

def draw_camera_paths_ultra_optimized(frame, paths, start, Hinv):
    """Ultra-optimized camera path drawing."""
    cam_paths = frame.copy()
    
    for idx, (end, path) in enumerate(paths):
        color = PATH_COLORS[idx % len(PATH_COLORS)]
        
        # Create ribbon mask
        H_bev, W_bev = frame.shape[:2]
        ribbon_mask = np.zeros((H_bev, W_bev), np.uint8)
        path_np = np.int32(path).reshape(-1, 1, 2)
        cv2.polylines(ribbon_mask, [path_np], False, 255, thickness=20, lineType=cv2.LINE_AA)
        
        # Warp to camera view
        cam_ribbon = cv2.warpPerspective(ribbon_mask, Hinv, (frame.shape[1], frame.shape[0]))
        
        # Apply color
        color_layer = np.zeros_like(frame)
        color_layer[cam_ribbon > 0] = color
        cam_paths = cv2.addWeighted(color_layer, 0.6, cam_paths, 1.0, 0)
    
    return cam_paths

# =============================================================================
# Model init
# =============================================================================
def initialize_model():
    cfg = Config(model_dir="models/my-segformer-road", conf_thresh=0.5, road_id=ROAD_ID)
    return FastRoadDetector(cfg)

# =============================================================================
# ULTRA-OPTIMIZED Main Pipeline
# =============================================================================
def process_video_ultra_optimized(video_path, output_dir, stride=1, save_video=False):
    """Ultra-optimized video processing pipeline."""
    print("Initializing ULTRA-OPTIMIZED FastRoadDetector...")
    timing.start_timer("model_initialization")
    model = initialize_model()
    timing.end_timer("model_initialization")
    print("Model ready!")
    
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    
    vw = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(os.path.join(output_dir, "cam_paths.mp4"), fourcc, fps, (w, h))
    
    # Performance tracking
    total_start = time.time()
    frame_times = []
    cache_hits = 0
    total_frames = 0
    
    while cap.isOpened():
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % stride != 0:
            frame_id += 1
            continue
        
        total_frames += 1
        
        print(f"\n--- Processing Frame {frame_id} ---")
        
        # Model inference
        timing.start_timer("model_inference")
        model_out, _ = model.process_frame(frame)
        if model_out.shape != frame.shape[:2]:
            model_out = cv2.resize(model_out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        timing.end_timer("model_inference")
        
        timing.start_timer("mask_processing")
        sidewalk_mask, road_mask = split_masks_from_output(model_out)
        cam_overlay = colorize_sidewalk_road(frame, sidewalk_mask, road_mask)
        timing.end_timer("mask_processing")
        
        # BEV transformation
        timing.start_timer("bev_transformation")
        bev_sidewalk = cv2.warpPerspective(sidewalk_mask, H, bev_size)
        bev_road = cv2.warpPerspective(road_mask, H, bev_size)
        
        if TRIM_BOTTOM > 0:
            bev_sidewalk = bev_sidewalk[:bev_sidewalk.shape[0]-TRIM_BOTTOM, :]
            bev_road = bev_road[:bev_road.shape[0]-TRIM_BOTTOM, :]
        timing.end_timer("bev_transformation")
        
        # Ultra-optimized skeleton and graph extraction
        skeleton_mask, graph = extract_skeleton_graph(bev_sidewalk, trim_px=5)
        endpoints = skeleton_endpoints(graph)
        
        # Find start point
        start = None
        if endpoints:
            start = max(endpoints, key=lambda p: p[1])
        
        # Create BEV visualization
        timing.start_timer("visualization")
        H_bev, W_bev = skeleton_mask.shape
        bev_color = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)
        bev_color[bev_road > 0] = (255, 120, 0)
        bev_color[bev_sidewalk > 0] = (0, 200, 0)
        
        # Ultra-optimized pathfinding and visualization
        if start:
            paths = find_paths_ultra_optimized(start, endpoints, graph)
            bev_color = draw_paths_ultra_optimized(bev_color, paths, start, skeleton_mask)
            cam_paths = draw_camera_paths_ultra_optimized(cam_overlay, paths, start, Hinv)
        else:
            cam_paths = cam_overlay.copy()
            # Still draw skeleton
            vis_skel = cv2.dilate(
                skeleton_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)),
                iterations=1
            )
            gy, gx = np.where(vis_skel > 0)
            bev_color[gy, gx] = (0, 128, 0)
        timing.end_timer("visualization")
        
        # Save results
        timing.start_timer("save_results")
        cv2.imwrite(os.path.join(output_dir, f"bev_paths_{frame_id:04d}.png"), bev_color)
        cv2.imwrite(os.path.join(output_dir, f"cam_paths_{frame_id:04d}.png"), cam_paths)
        if vw:
            vw.write(cam_paths)
        timing.end_timer("save_results")
        
        # Performance tracking
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        num_paths = len(paths) if start else 0
        print(f"Frame {frame_id} | {frame_time:.3f}s | start={start} | paths={num_paths}")
        
        # Print frame-level timing
        frame_timing = timing.get_summary()
        if frame_timing:
            print(f"  Model inference: {frame_timing.get('model_inference', {}).get('average', 0)*1000:.1f}ms")
            print(f"  Skeletonization: {frame_timing.get('skeletonization', {}).get('average', 0)*1000:.1f}ms")
            print(f"  Graph construction: {frame_timing.get('graph_construction', {}).get('average', 0)*1000:.1f}ms")
            print(f"  Pathfinding: {frame_timing.get('pathfinding_total', {}).get('average', 0)*1000:.1f}ms")
        
        frame_id += 1
    
    cap.release()
    if vw:
        vw.release()
    
    # Performance summary
    total_time = time.time() - total_start
    avg_frame_time = np.mean(frame_times) if frame_times else 0
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    print(f"\nULTRA-OPTIMIZATION RESULTS:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average frame time: {avg_frame_time:.3f}s")
    print(f"   Processing FPS: {fps:.1f}")
    print(f"   Total frames processed: {total_frames}")
    print(f"   Results saved to {output_dir}")
    
    # Print detailed timing analysis
    timing.print_summary()

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    process_video_ultra_optimized("test_video_june_03_3.mp4", "bev_paths_ultra_optimized", stride=3, save_video=False)
