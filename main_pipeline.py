import cv2
import sys
import os
from tqdm import tqdm

from src.detection.detector import VehicleDetector
from src.tracking.tracker import VehicleTracker
from src.lane_logic.lane_assigner import LaneAssigner
from src.analytics.logger import TrafficLogger

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH    = "data/raw_videos/traffic.mp4"
LANE_CONFIG   = "data/lane_config.json"
OUTPUT_VIDEO  = "outputs/annotated_output.mp4"
LOG_PATH      = "outputs/logs/traffic_log.csv"
PROCESS_EVERY = 2    # process every 2nd frame to save time
MAX_FRAMES    = 900  # process 900 frames first as a test (~30 seconds of video)
# ─────────────────────────────────────────────────────────────────────────────

def draw_vehicles(frame, vehicles):
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        is_violation    = v.get("violation", False)
        lane_id         = v.get("lane_id", "?")
        category        = v.get("category", "?")
        track_id        = v["track_id"]

        # Red box for violation, green for OK
        color     = (0, 0, 255) if is_violation else (0, 200, 100)
        thickness = 3 if is_violation else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{track_id} {category} L{lane_id}"
        if is_violation:
            label += " VIOLATION"

        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Dot at centroid
        if "centroid" in v:
            cv2.circle(frame, v["centroid"], 4, color, -1)

    return frame


def draw_stats(frame, vehicles, frame_id):
    total      = len(vehicles)
    violations = sum(1 for v in vehicles if v.get("violation"))

    # Black background panel
    cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)

    cv2.putText(frame, f"Frame     : {frame_id}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"Vehicles  : {total}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,100), 1)
    cv2.putText(frame, f"Violations: {violations}",
                (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    return frame


def run():
    print("\n" + "="*55)
    print("  INDIA LANE ANALYTICS — MAIN PIPELINE")
    print("="*55)

    # Check all required files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        return
    if not os.path.exists(LANE_CONFIG):
        print(f"ERROR: Lane config not found at {LANE_CONFIG}")
        print("Run draw_lanes.py first.")
        return

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # Initialize all components
    print("\nLoading components...")
    detector = VehicleDetector()
    tracker  = VehicleTracker()
    assigner = LaneAssigner(LANE_CONFIG)
    logger   = TrafficLogger(LOG_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / PROCESS_EVERY,
        (w, h)
    )

    print(f"\nProcessing first {MAX_FRAMES} frames...")
    print(f"Video : {w}x{h} @ {fps}fps")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Log   : {LOG_PATH}\n")

    frame_id         = 0
    processed        = 0
    total_violations = 0

    with tqdm(total=MAX_FRAMES, unit="frames") as pbar:
        while cap.isOpened() and frame_id < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Skip every other frame
            if frame_id % PROCESS_EVERY != 0:
                continue

            # ── THE PIPELINE ──────────────────────────────────────
            detections = detector.detect(frame)
            tracked    = tracker.update(detections, frame)
            assigned   = assigner.assign(tracked)
            logger.log(frame_id, assigned)
            # ──────────────────────────────────────────────────────

            # Draw output
            frame = assigner.draw_lanes(frame)
            frame = draw_vehicles(frame, assigned)
            frame = draw_stats(frame, assigned, frame_id)
            out.write(frame)

            violations    = sum(1 for v in assigned if v.get("violation"))
            total_violations += violations
            processed += 1
            pbar.update(PROCESS_EVERY)
            pbar.set_postfix({
                "vehicles": len(assigned),
                "violations": violations
            })

    cap.release()
    out.release()

    print("\n" + "="*55)
    print("  PIPELINE COMPLETE")
    print("="*55)
    print(f"  Frames processed  : {processed}")
    print(f"  Total violations  : {total_violations}")
    print(f"  Annotated video   : {OUTPUT_VIDEO}")
    print(f"  Traffic data log  : {LOG_PATH}")
    print("="*55)
    print("\nNext step: run  python analytics.py  to see stats")


if __name__ == "__main__":
    run()