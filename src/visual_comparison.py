#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from fast_road_detector import FastRoadDetector, Config as FastConfig
from advanced_road_detector import AdvancedRoadDetector, AdvancedConfig, Approach

class VisualComparison:
    def __init__(self, video_path: str, model_dir: str = "models/my-segformer-road_new"):
        """Initialize the visual comparison framework."""
        self.video_path = video_path
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = os.path.join("comparison_results", "visual")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize detectors
        self._create_detectors()
        
    def _create_detectors(self):
        """Create instances of all detectors."""
        # Basic detector
        processor = AutoImageProcessor.from_pretrained(self.model_dir)
        model = SegformerForSemanticSegmentation.from_pretrained(self.model_dir).to(self.device)
        model.eval()
        self.basic_detector = (model, processor)
        
        # Fast detector
        fast_config = FastConfig(
            video_path=self.video_path,
            model_dir=self.model_dir,
            output_mp4="result/fast_overlay.mp4"
        )
        self.fast_detector = FastRoadDetector(fast_config)
        
        # Advanced detectors
        self.advanced_detectors = {}
        for approach in Approach:
            config = AdvancedConfig(
                video_path=self.video_path,
                output_mp4=f"result/advanced_{approach.value}_overlay.mp4",
                model_dir=self.model_dir,
                approach=approach
            )
            self.advanced_detectors[approach.name] = AdvancedRoadDetector(config)
        
    def _process_frame_basic(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using basic detector."""
        model, processor = self.basic_detector
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # Resize to original size
        up = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False
        )[0]
        
        # Get road probabilities
        probs = up.softmax(0)[1].cpu().numpy()  # road_id = 1
        mask = (probs > 0.4).astype(np.uint8) * 255
        
        # Convert to RGB for visualization
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask_rgb
        
    def _process_frame_fast(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using fast detector."""
        detector = self.fast_detector
        mask, _ = detector.process_frame(frame)  # Unpack the tuple, ignore the frame
        
        # Convert to RGB for visualization
        if len(mask.shape) == 2:  # If grayscale
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = mask
        return mask_rgb
        
    def _process_frame_advanced(self, frame: np.ndarray) -> dict:
        """Process frame using all advanced detector approaches."""
        results = {}
        for approach in self.advanced_detectors.keys():
            detector = self.advanced_detectors[approach]
            mask = detector.process_frame(frame)
            
            # Convert to RGB for visualization
            if len(mask.shape) == 2:  # If grayscale
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            else:
                mask_rgb = mask
            results[f"Advanced_{approach}"] = mask_rgb
        return results

    def _load_ground_truth(self, frame_idx: int, target_size: tuple) -> np.ndarray:
        """Load ground truth mask for a given frame and resize to target size."""
        mask_path = os.path.join("data", "masks", f"frame_{frame_idx:04d}.jpg")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize to match input frame size
                mask = cv2.resize(mask, (target_size[1], target_size[0]))
                # Convert to binary mask (0 or 255)
                mask = (mask > 127).astype(np.uint8) * 255
                return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return None

    def _create_comparison_image(self, frame: np.ndarray, results: dict, frame_idx: int):
        """Create a comparison image with all segmentation results."""
        # Calculate grid dimensions
        n_detectors = len(results)
        grid_size = int(np.ceil(np.sqrt(n_detectors + 1)))  # +1 for original frame
        
        # Create figure
        plt.figure(figsize=(20, 20))
        
        # Plot original frame
        plt.subplot(grid_size, grid_size, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Original Frame")
        plt.axis('off')
        
        # Plot segmentation results
        for idx, (name, mask) in enumerate(results.items(), start=2):
            plt.subplot(grid_size, grid_size, idx)
            plt.imshow(mask)  # mask is already in RGB format
            plt.title(f"{name}\nRoad Pixels: {np.sum(mask[:,:,0] > 0) / (mask.shape[0] * mask.shape[1]):.2%}")
            plt.axis('off')
            
        # Save comparison
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f"comparison_frame_{frame_idx:04d}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
        
    def generate_comparisons(self, num_frames: int = 5):
        """Generate visual comparisons for evenly spaced frames across the video."""
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames > total_frames:
            num_frames = total_frames
        # Calculate evenly spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        saved_frame_indices = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # Process frame with all detectors
            results = {
                "Basic": self._process_frame_basic(frame),
                "Fast": self._process_frame_fast(frame)
            }
            # Add all advanced detector results
            results.update(self._process_frame_advanced(frame))
            # Create and save comparison
            self._create_comparison_image(frame, results, frame_idx)
            saved_frame_indices.append(frame_idx)
        cap.release()
        self._generate_html_report(saved_frame_indices)
        
    def _generate_html_report(self, frame_indices: list):
        """Create an HTML report with all comparisons."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Road Detection Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .comparison { margin-bottom: 30px; }
                h2 { color: #333; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Road Detection Comparison</h1>
        """
        
        for frame_idx in frame_indices:
            html_content += f"""
            <div class="comparison">
                <h2>Frame {frame_idx}</h2>
                <img src="comparison_frame_{frame_idx:04d}.png" alt="Comparison for frame {frame_idx}">
            </div>
            """
            
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(self.output_dir, "visual_comparison_report.html"), "w") as f:
            f.write(html_content)

def main():
    """Main function to run the visual comparison."""
    # Create visual comparison
    comparison = VisualComparison(
        video_path="test_video_june_03_3.mp4",
        model_dir="models/my-segformer-road_new"
    )
    
    # Generate comparisons
    comparison.generate_comparisons(num_frames=5)

if __name__ == "__main__":
    main() 