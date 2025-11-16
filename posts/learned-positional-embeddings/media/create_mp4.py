import cv2
import glob
import re

# Collect and numerically sort files
files = glob.glob("self_similarity_during_training/positional_self_similarity_learned_step*.png")[:60]

def extract_step(fname):
    m = re.search(r"step(\d+)", fname)
    return int(m.group(1)) if m else -1

files = sorted(files, key=extract_step)

# Load first frame to compute target size
first = cv2.imread(files[0])
h, w, _ = first.shape
target_w = 500
scale = target_w / w
target_h = int(h * scale)

# H.264 for Chrome compatibility
fourcc = cv2.VideoWriter_fourcc(*"avc1")
video = cv2.VideoWriter("output.mp4", fourcc, 10, (target_w, target_h))

for f in files:
    img = cv2.imread(f)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    video.write(resized)

video.release()
