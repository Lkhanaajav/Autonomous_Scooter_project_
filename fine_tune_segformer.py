#!/usr/bin/env python3
import os
import time
import cv2
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

@dataclass
class Config:
    video_path: str = "test_video_june_03_1.MOV"
    model_dir: str = "models/my-segformer-road_new"
    output_mp4: str = "result/overlay_edge_cleaned.mp4"
    road_id: int = 1
    conf_thresh: float = 0.6
    frame_step: int = 5
    batch_size: int = 4
    use_gpu: bool = True
    enable_logging: bool = True

class RoadDetector:
    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self._setup_device()
        self._load_model()
        
    def _setup_logging(self):
        if self.config.enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())

    def _setup_device(self):
        self.device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self):
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.config.model_dir)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.config.model_dir
            ).to(self.device)
            self.model.eval()
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single frame and return the segmentation mask and overlay."""
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        inputs = self.processor(rgb, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Process predictions
        H, W = frame.shape[:2]
        up = torch.nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )[0]
        probs = up.softmax(0)[self.config.road_id].cpu().numpy()
        mask = (probs > self.config.conf_thresh).astype(np.uint8) * 255
        
        # Edge-based cleaning
        mask = self._clean_mask(mask, frame)
        
        # Create overlay
        overlay = frame.copy()
        overlay[mask == 255] = (0, 255, 0)
        
        return mask, overlay

    def _clean_mask(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Clean the segmentation mask using edge detection."""
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 1.5)
        edges = cv2.Canny(blur, 50, 150)
        barrier = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        
        # Remove edge crossings
        mask[barrier > 0] = 0
        
        # Remove small islands
        num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_lbl > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + np.argmax(areas)
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)
            
        return mask

    def process_video(self):
        """Process the entire video with the configured settings."""
        # Setup video capture
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        os.makedirs(os.path.dirname(self.config.output_mp4), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vis = cv2.VideoWriter(self.config.output_mp4, fourcc, fps, (W, H), isColor=True)

        # Processing loop
        frame_idx = 0
        processed_idx = 0
        prev_mask = None
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.config.frame_step == 0:
                    mask, overlay = self.process_frame(frame)
                    prev_mask = mask
                    processed_idx += 1
                else:
                    overlay = frame.copy()
                    if prev_mask is not None:
                        overlay[prev_mask == 255] = (0, 255, 0)

                # Preview
                preview = cv2.resize(overlay, (W//2, H//2))
                cv2.imshow("Edge-Cleaned Overlay", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                out_vis.write(overlay)
                frame_idx += 1

        finally:
            # Cleanup
            cap.release()
            out_vis.release()
            cv2.destroyAllWindows()

        # Log statistics
        elapsed = time.time() - t0
        self.logger.info(f"Processed {frame_idx} frames, ran inference on {processed_idx} frames")
        self.logger.info(f"Processing time: {elapsed:.2f}s → {frame_idx/elapsed:.2f} FPS")
        self.logger.info(f"Output saved to {self.config.output_mp4}")

def main():
    config = Config()
    detector = RoadDetector(config)
    detector.process_video()

if __name__ == "__main__":
    main()
