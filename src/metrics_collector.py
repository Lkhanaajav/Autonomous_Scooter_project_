#!/usr/bin/env python3
import os
import time
import psutil
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
from datetime import datetime

@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enable_gpu_metrics: bool = True
    enable_cpu_metrics: bool = True
    enable_memory_metrics: bool = True
    enable_timing_metrics: bool = True
    enable_accuracy_metrics: bool = True
    log_interval: int = 100  # Log metrics every N frames
    save_interval: int = 1000  # Save metrics every N frames

@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_number: int
    processing_time: float  # ms
    gpu_memory_allocated: float  # MB
    gpu_memory_reserved: float  # MB
    cpu_memory_used: float  # MB
    inference_time: float  # ms
    post_processing_time: float  # ms
    accuracy: float  # Placeholder for accuracy metric

@dataclass
class AggregateMetrics:
    """Aggregated metrics over multiple frames."""
    total_frames: int
    processed_frames: int
    total_time: float  # seconds
    average_fps: float
    average_inference_time: float  # ms
    average_processing_time: float  # ms
    peak_gpu_memory: float  # MB
    peak_cpu_memory: float  # MB
    average_accuracy: float
    frame_metrics: List[FrameMetrics]

class MetricsCollector:
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.frame_metrics: List[FrameMetrics] = []
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_save_time = self.start_time
        
    def start_frame(self) -> float:
        """Start timing a frame."""
        return time.time()
        
    def end_frame(self, start_time: float, frame_number: int) -> FrameMetrics:
        """End timing a frame and collect metrics."""
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Collect memory metrics
        gpu_allocated = 0.0
        gpu_reserved = 0.0
        if self.config.enable_gpu_metrics and torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_reserved = torch.cuda.memory_reserved() / 1024**2
            
        cpu_used = 0.0
        if self.config.enable_cpu_metrics:
            cpu_used = psutil.Process().memory_info().rss / 1024**2
            
        # Create frame metrics
        metrics = FrameMetrics(
            frame_number=frame_number,
            processing_time=processing_time,
            gpu_memory_allocated=gpu_allocated,
            gpu_memory_reserved=gpu_reserved,
            cpu_memory_used=cpu_used,
            inference_time=0.0,  # To be set by the detector
            post_processing_time=0.0,  # To be set by the detector
            accuracy=0.0  # To be set by the detector
        )
        
        self.frame_metrics.append(metrics)
        
        # Log metrics if interval reached
        if len(self.frame_metrics) % self.config.log_interval == 0:
            self._log_metrics()
            
        # Save metrics if interval reached
        if len(self.frame_metrics) % self.config.save_interval == 0:
            self._save_metrics()
            
        return metrics
        
    def _log_metrics(self):
        """Log current metrics."""
        if not self.frame_metrics:
            return
            
        latest = self.frame_metrics[-1]
        print(f"\nMetrics at frame {latest.frame_number}:")
        print(f"Processing time: {latest.processing_time:.2f} ms")
        if self.config.enable_gpu_metrics:
            print(f"GPU Memory: {latest.gpu_memory_allocated:.2f} MB allocated, {latest.gpu_memory_reserved:.2f} MB reserved")
        if self.config.enable_cpu_metrics:
            print(f"CPU Memory: {latest.cpu_memory_used:.2f} MB")
        print(f"Inference time: {latest.inference_time:.2f} ms")
        print(f"Post-processing time: {latest.post_processing_time:.2f} ms")
        if self.config.enable_accuracy_metrics:
            print(f"Accuracy: {latest.accuracy:.4f}")
            
    def _save_metrics(self):
        """Save current metrics to file."""
        if not self.frame_metrics:
            return
            
        # Create metrics directory if it doesn't exist
        os.makedirs("metrics", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics/metrics_{timestamp}.json"
        
        # Calculate aggregate metrics
        aggregate = self.get_aggregate_metrics()
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump({
                "config": asdict(self.config),
                "aggregate_metrics": asdict(aggregate),
                "frame_metrics": [asdict(m) for m in self.frame_metrics]
            }, f, indent=2)
            
    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Calculate aggregate metrics over all frames."""
        if not self.frame_metrics:
            return AggregateMetrics(
                total_frames=0,
                processed_frames=0,
                total_time=0.0,
                average_fps=0.0,
                average_inference_time=0.0,
                average_processing_time=0.0,
                peak_gpu_memory=0.0,
                peak_cpu_memory=0.0,
                average_accuracy=0.0,
                frame_metrics=self.frame_metrics
            )
            
        total_time = time.time() - self.start_time
        processed_frames = len(self.frame_metrics)
        
        # Calculate averages
        avg_inference = np.mean([m.inference_time for m in self.frame_metrics])
        avg_processing = np.mean([m.processing_time for m in self.frame_metrics])
        avg_accuracy = np.mean([m.accuracy for m in self.frame_metrics])
        
        # Calculate peaks
        peak_gpu = max([m.gpu_memory_allocated for m in self.frame_metrics])
        peak_cpu = max([m.cpu_memory_used for m in self.frame_metrics])
        
        return AggregateMetrics(
            total_frames=self.frame_metrics[-1].frame_number + 1,
            processed_frames=processed_frames,
            total_time=total_time,
            average_fps=processed_frames / total_time if total_time > 0 else 0,
            average_inference_time=avg_inference,
            average_processing_time=avg_processing,
            peak_gpu_memory=peak_gpu,
            peak_cpu_memory=peak_cpu,
            average_accuracy=avg_accuracy,
            frame_metrics=self.frame_metrics
        )
        
    def finalize(self):
        """Finalize metrics collection and save final results."""
        self._save_metrics()
        aggregate = self.get_aggregate_metrics()
        
        print("\nFinal Metrics Summary:")
        print(f"Total frames: {aggregate.total_frames}")
        print(f"Processed frames: {aggregate.processed_frames}")
        print(f"Total time: {aggregate.total_time:.2f} seconds")
        print(f"Average FPS: {aggregate.average_fps:.2f}")
        print(f"Average inference time: {aggregate.average_inference_time:.2f} ms")
        print(f"Average processing time: {aggregate.average_processing_time:.2f} ms")
        print(f"Peak GPU memory: {aggregate.peak_gpu_memory:.2f} MB")
        print(f"Peak CPU memory: {aggregate.peak_cpu_memory:.2f} MB")
        if self.config.enable_accuracy_metrics:
            print(f"Average accuracy: {aggregate.average_accuracy:.4f}") 