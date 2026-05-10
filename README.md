# Autonomous Scooter Navigation System

Real-time autonomous scooter navigation using camera feeds, semantic segmentation with SegFormer, bird's-eye-view transformation, and Dijkstra path planning.

## Overview

This project delivers an end-to-end autonomous navigation pipeline for a scooter platform. A live camera feed is processed through a deep-learning semantic segmentation model (SegFormer), the segmented road mask is warped into a bird's-eye-view, and Dijkstra's algorithm computes a collision-free path to the next waypoint -- all in real time.

## Tech Stack

- **Language:** Python
- **Segmentation:** SegFormer (Hugging Face Transformers)
- **Path Planning:** Dijkstra's algorithm
- **Perception:** Bird's-eye-view (BEV) transformation
- **Camera Pipeline:** fast_camera.py, camera_waypoint_pipeline.py

## Key Features

- **Real-time road segmentation** - SegFormer classifies drivable vs. non-drivable pixels at low latency directly from the scooter's camera
- **BEV path visualization** - perspective warp converts segmented frames into a metrically accurate top-down view for path planning
- **Waypoint pipeline** - camera_waypoint_pipeline.py chains perception, BEV projection, and planning into a single streaming loop
- **Dijkstra path planning** - guaranteed shortest-path routing on the BEV occupancy grid

## Repository Structure

`
Autonomous_Scooter_project_/
+-- camera_waypoint_pipeline.py   # End-to-end perception-to-waypoint loop
+-- fast_camera.py                # Low-latency camera capture module
+-- bev_paths_dijkstra.py         # BEV grid construction + Dijkstra planner
+-- segmentation/                 # SegFormer output sample images
`

## What Was Interesting

The most compelling challenge was bridging the gap between deep learning and classical planning: SegFormer produces probabilistic pixel labels, but Dijkstra needs a crisp binary occupancy grid. Tuning the confidence threshold and morphological post-processing to handle shadows, lighting changes, and partial occlusions -- while keeping the pipeline fast enough for real-time control -- required careful co-design of the perception and planning stages.

## License

MIT