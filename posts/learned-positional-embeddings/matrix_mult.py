import cv2

frames = [f"positional_self_similarity_learned_epoch{i}.png" for i in range(1, 100)]

# Load first frame to compute target height
first = cv2.imread(frames[0])
h, w, _ = first.shape
target_w = 500
scale = target_w / w
target_h = int(h * scale)

# Video writer (MP4, 30 fps)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("output.mp4", fourcc, 30, (target_w, target_h))

for f in frames:
    img = cv2.imread(f)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    video.write(resized)

video.release()
