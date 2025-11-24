import cv2
import glob
import re
import os

# Helper to extract numeric step from filename
def extract_step(fname):
    m = re.search(r"step(\d+)", fname)
    return int(m.group(1)) if m else -1

# Collect all matching files once
all_files = glob.glob("training_snapshots/learned_step*.png")

# Split into "normal" and "_values" variants
base_files = [f for f in all_files if ("_values" not in os.path.basename(f) and "_density" not in os.path.basename(f))]
values_files = [f for f in all_files if "_values" in os.path.basename(f)]
density_files = [f for f in all_files if "_density" in os.path.basename(f)]

print(len(base_files))

# Sort numerically by step and optionally limit
base_files = sorted(base_files, key=extract_step)
values_files = sorted(values_files, key=extract_step)
density_files = sorted(density_files, key=extract_step)

if not base_files and not values_files:
    raise RuntimeError("No matching images found.")

# Determine target size from first available frame
first_file = base_files[0] if base_files else values_files[0]
first_img = cv2.imread(first_file)
if first_img is None:
    raise RuntimeError(f"Failed to read first image: {first_file}")

h, w, _ = first_img.shape
target_w = 500
scale = target_w / w
target_h = int(h * scale)

# H.264 for Chrome compatibility
fourcc = cv2.VideoWriter_fourcc(*"avc1")

# Create writers (only if we have frames for that video)
base_video = cv2.VideoWriter("output_self_sim.mp4", fourcc, 30, (target_w, target_h)) if base_files else None
values_video = cv2.VideoWriter("output_values.mp4", fourcc, 30, (target_w, target_h)) if values_files else None

for f in base_files:
    img = cv2.imread(f)
    if img is None:
        continue
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    base_video.write(resized)
base_video.release()

for f in values_files:
    img = cv2.imread(f)
    if img is None:
        continue
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    values_video.write(resized)
values_video.release()

# Determine target size from first available frame
first_file = density_files[0]
first_img = cv2.imread(first_file)
if first_img is None:
    raise RuntimeError(f"Failed to read first image: {first_file}")

h, w, _ = first_img.shape
target_w = 600
scale = target_w / w
target_h = int(h * scale)

density_video = cv2.VideoWriter("output_density.mp4", fourcc, 20, (target_w, target_h)) if density_files else None



    
for f in density_files:
    img = cv2.imread(f)
    if img is None:
        continue
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    density_video.write(resized)
density_video.release()

