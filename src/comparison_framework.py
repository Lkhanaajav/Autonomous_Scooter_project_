#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

@dataclass
class ComparisonConfig:
    """Configuration for comparison analysis."""
    metrics_dir: str = "metrics"
    output_dir: str = "comparison_results"
    plot_format: str = "png"
    dpi: int = 300
    show_plots: bool = True
    save_plots: bool = True

class ComparisonFramework:
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self._setup_output_dir()
        
    def _setup_output_dir(self):
        """Create output directory for comparison results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def load_metrics(self, metrics_file: str) -> List[Dict]:
        """Load metrics from a JSON file."""
        with open(metrics_file, 'r') as f:
            return json.load(f)
            
    def compare_detectors(self, metrics_files: List[str]):
        """Compare multiple detector implementations."""
        # Load all metrics
        all_metrics = []
        for f in metrics_files:
            metrics = self.load_metrics(f)
            all_metrics.extend(metrics)
        
        # Create comparison tables
        self._create_performance_table(all_metrics)
        self._create_memory_table(all_metrics)
        
        # Generate comparison plots
        self._plot_fps_comparison(all_metrics)
        self._plot_memory_comparison(all_metrics)
        self._plot_processing_time_comparison(all_metrics)
        
    def _create_performance_table(self, metrics_data: List[Dict]):
        """Create a table comparing performance metrics."""
        # Extract relevant metrics
        performance_data = []
        for data in metrics_data:
            metrics = data['metrics']
            config = data['config']
            # Calculate theoretical frame count based on duration and frame step
            theoretical_frames = int(metrics['total_time'] * 30 / config['frame_step'])  # Assuming 30 FPS video
            
            # Calculate actual FPS based on processed frames
            actual_fps = metrics['processed_count'] / metrics['total_time']
            
            performance_data.append({
                'Detector': data['config'].get('detector_type', 'Unknown'),
                'Approach': data['config'].get('approach', 'baseline'),
                'FPS': actual_fps,  # Use actual FPS based on processed frames
                'Inference Time (ms)': metrics['inference_time'],
                'Total Time (s)': metrics['total_time'],
                'Theoretical Frames': theoretical_frames,
                'Processed Frames': metrics['processed_count'],
                'Processing Overhead (%)': ((metrics['total_time'] * 1000) / (metrics['inference_time'] * metrics['processed_count']) - 1) * 100
            })
            
        # Create DataFrame
        df = pd.DataFrame(performance_data)
        
        # Calculate averages for each detector/approach combination
        df_avg = df.groupby(['Detector', 'Approach']).mean().reset_index()
        
        # Round numeric columns
        numeric_cols = ['FPS', 'Inference Time (ms)', 'Total Time (s)', 'Processing Overhead (%)']
        df_avg[numeric_cols] = df_avg[numeric_cols].round(2)
        
        # Save to CSV
        output_file = os.path.join(self.config.output_dir, 'performance_comparison.csv')
        df_avg.to_csv(output_file, index=False)
        
        # Print table
        print("\nPerformance Comparison:")
        print(df_avg.to_string(index=False))
        print("\nNote: FPS is calculated based on processed frames, not total video frames")
        print("Processing Overhead (%) shows the additional time spent on processing techniques beyond model inference")
        
    def _create_memory_table(self, metrics_data: List[Dict]):
        """Create a table comparing memory usage."""
        # Extract relevant metrics
        memory_data = []
        for data in metrics_data:
            metrics = data['metrics']
            memory_usage = metrics['memory_usage']
            memory_data.append({
                'Detector': data['config'].get('detector_type', 'Unknown'),
                'Approach': data['config'].get('approach', 'baseline'),
                'Peak CPU Memory (MB)': memory_usage.get('cpu', 0),
                'Peak GPU Memory (MB)': memory_usage.get('gpu', 0)
            })
            
        # Create DataFrame
        df = pd.DataFrame(memory_data)
        
        # Calculate averages for each detector/approach combination
        df_avg = df.groupby(['Detector', 'Approach']).mean().reset_index()
        
        # Save to CSV
        output_file = os.path.join(self.config.output_dir, 'memory_comparison.csv')
        df_avg.to_csv(output_file, index=False)
        
        # Print table
        print("\nMemory Usage Comparison:")
        print(df_avg.to_string(index=False))
        
    def _plot_fps_comparison(self, metrics_data: List[Dict]):
        """Create a bar plot comparing FPS across detectors."""
        plt.figure(figsize=(10, 6))
        
        # Extract and average data
        fps_data = {}
        for data in metrics_data:
            detector = data['config'].get('detector_type', 'Unknown')
            approach = data['config'].get('approach', 'baseline')
            key = f"{detector}\n({approach})"
            if key not in fps_data:
                fps_data[key] = []
            fps_data[key].append(data['metrics']['fps'])
            
        # Calculate averages
        detectors = list(fps_data.keys())
        fps_values = [np.mean(fps_data[d]) for d in detectors]
            
        # Create bar plot
        plt.bar(detectors, fps_values)
        plt.title('FPS Comparison Across Detectors')
        plt.xlabel('Detector Type')
        plt.ylabel('Frames Per Second')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        if self.config.save_plots:
            output_file = os.path.join(self.config.output_dir, 'fps_comparison.png')
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            
        # Show plot
        if self.config.show_plots:
            plt.show()
            
        plt.close()
        
    def _plot_memory_comparison(self, metrics_data: List[Dict]):
        """Create a bar plot comparing memory usage."""
        plt.figure(figsize=(12, 6))
        
        # Extract and average data
        memory_data = {}
        for data in metrics_data:
            detector = data['config'].get('detector_type', 'Unknown')
            approach = data['config'].get('approach', 'baseline')
            key = f"{detector}\n({approach})"
            if key not in memory_data:
                memory_data[key] = {'gpu': [], 'cpu': []}
            memory_usage = data['metrics']['memory_usage']
            memory_data[key]['gpu'].append(memory_usage.get('gpu', 0))
            memory_data[key]['cpu'].append(memory_usage.get('cpu', 0))
            
        # Calculate averages
        detectors = list(memory_data.keys())
        gpu_memory = [np.mean(memory_data[d]['gpu']) for d in detectors]
        cpu_memory = [np.mean(memory_data[d]['cpu']) for d in detectors]
            
        # Create grouped bar plot
        x = np.arange(len(detectors))
        width = 0.35
        
        plt.bar(x - width/2, gpu_memory, width, label='GPU Memory')
        plt.bar(x + width/2, cpu_memory, width, label='CPU Memory')
        
        plt.title('Memory Usage Comparison')
        plt.xlabel('Detector Type')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(x, detectors, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        if self.config.save_plots:
            output_file = os.path.join(self.config.output_dir, 'memory_comparison.png')
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            
        # Show plot
        if self.config.show_plots:
            plt.show()
            
        plt.close()
        
    def _plot_processing_time_comparison(self, metrics_data: List[Dict]):
        """Create a bar plot comparing processing times."""
        plt.figure(figsize=(12, 6))
        
        # Extract and average data
        time_data = {}
        for data in metrics_data:
            detector = data['config'].get('detector_type', 'Unknown')
            approach = data['config'].get('approach', 'baseline')
            key = f"{detector}\n({approach})"
            if key not in time_data:
                time_data[key] = {'inference': [], 'total': []}
            time_data[key]['inference'].append(data['metrics']['inference_time'])
            time_data[key]['total'].append(data['metrics']['total_time'] * 1000)  # Convert to ms
            
        # Calculate averages
        detectors = list(time_data.keys())
        inference_time = [np.mean(time_data[d]['inference']) for d in detectors]
        total_time = [np.mean(time_data[d]['total']) for d in detectors]
            
        # Create grouped bar plot
        x = np.arange(len(detectors))
        width = 0.35
        
        plt.bar(x - width/2, inference_time, width, label='Inference Time')
        plt.bar(x + width/2, total_time, width, label='Total Time')
        
        plt.title('Processing Time Comparison')
        plt.xlabel('Detector Type')
        plt.ylabel('Time (ms)')
        plt.xticks(x, detectors, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        if self.config.save_plots:
            output_file = os.path.join(self.config.output_dir, 'processing_time_comparison.png')
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            
        # Show plot
        if self.config.show_plots:
            plt.show()
            
        plt.close()
        
    def generate_comparison_report(self, metrics_files: List[str]):
        """Generate a comprehensive comparison report."""
        # Create comparison tables and plots
        self.compare_detectors(metrics_files)
        
        # Generate HTML report
        self._generate_html_report(metrics_files)
        
    def _generate_html_report(self, metrics_files: List[str]):
        """Generate an HTML report with all comparisons."""
        html_content = """
        <html>
        <head>
            <title>Road Detector Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Road Detector Comparison Report</h1>
            <h2>Performance Comparison</h2>
            <img src="performance_comparison.png" alt="Performance Comparison">
            <h2>Memory Usage Comparison</h2>
            <img src="memory_comparison.png" alt="Memory Usage Comparison">
            <h2>Processing Time Comparison</h2>
            <img src="processing_time_comparison.png" alt="Processing Time Comparison">
        </body>
        </html>
        """
        
        # Save HTML report
        output_file = os.path.join(self.config.output_dir, 'comparison_report.html')
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        print(f"\nComparison report generated: {output_file}")

def main():
    # Create comparison configuration
    config = ComparisonConfig()
    
    # Get latest metrics file
    metrics_dir = Path(config.metrics_dir)
    metrics_files = sorted(metrics_dir.glob('metrics_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not metrics_files:
        print("No metrics files found!")
        return
        
    # Create comparison framework
    framework = ComparisonFramework(config)
    
    # Generate comparison report
    framework.generate_comparison_report([str(f) for f in metrics_files[:1]])  # Use only the latest file

if __name__ == "__main__":
    main() 