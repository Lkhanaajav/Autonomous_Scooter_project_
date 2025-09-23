import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Paths (modify if your data is in a different directory)
IMAGE_DIR = os.path.join('..', 'data', 'preview_frames')  # input images folder
JSON_PATH = os.path.join('..', 'data', 'makesense.json')  # hand-annotation JSON
OUT_DIR   = os.path.join('..', 'data', 'masks_hand')      # output masks folder

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load the JSON file
with open(JSON_PATH, 'r') as f:
    annotations = json.load(f)

# JSON is a dict keyed by filename
items = annotations.items()

# Process each annotated image
for filename, record in tqdm(items, desc='Rasterizing hand masks'):
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not load image {filename}")
        continue
    height, width = img.shape[:2]

    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill 'road' polygons
    regions = record.get('regions', {})
    for region in regions.values():
        attrs = region.get('region_attributes', {})
        if attrs.get('label') != 'road':
            continue
        shape = region.get('shape_attributes', {})
        if shape.get('name') != 'polygon':
            continue
        pts = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Save the mask PNG
    base = os.path.splitext(filename)[0]
    out_path = os.path.join(OUT_DIR, f"{base}.png")
    cv2.imwrite(out_path, mask)

print(f"\n✅ Completed: Saved hand masks to '{OUT_DIR}'")