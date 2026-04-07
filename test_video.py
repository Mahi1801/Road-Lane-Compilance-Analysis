import cv2
import os

VIDEO_PATH = "data/raw_videos/traffic.mp4"

# Check if file exists
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video not found at {VIDEO_PATH}")
    print("Please place your traffic video at that path and try again.")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print("ERROR: Could not open video file.")
    exit()

fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps if fps > 0 else 0

print("=" * 40)
print("VIDEO PROPERTIES")
print("=" * 40)
print(f"FPS          : {fps}")
print(f"Total Frames : {total_frames}")
print(f"Width        : {width} px")
print(f"Height       : {height} px")
print(f"Duration     : {duration_sec:.1f} seconds ({duration_sec/60:.1f} minutes)")
print("=" * 40)

# Save the first frame as an image
os.makedirs("outputs", exist_ok=True)
ret, frame = cap.read()

if ret:
    cv2.imwrite("outputs/first_frame.jpg", frame)
    print("First frame saved -> outputs/first_frame.jpg")
    print("Open that image to see your road frame.")
    print("You will draw lane boundaries on this image in the next step.")
else:
    print("ERROR: Could not read first frame from video.")

cap.release()
print("\nDone!")