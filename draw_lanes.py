import cv2
import json
import numpy as np
import os

IMAGE_PATH  = "outputs/first_frame.jpg"
OUTPUT_PATH = "data/lane_config.json"

# ── Load the road frame ──────────────────────────────────────────────────────
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("ERROR: Could not load outputs/first_frame.jpg")
    print("Run test_video.py first.")
    exit()

clone = frame.copy()

# ── Lane definitions ─────────────────────────────────────────────────────────
LANE_LABELS = {
    1: "Lane1-Cyclist",
    2: "Lane2-Bikes",
    3: "Lane3-Cars",
    4: "Lane4-Trucks"
}
LANE_COLORS = {
    1: (0, 255, 0),      # green
    2: (0, 165, 255),    # orange
    3: (255, 100, 0),    # blue
    4: (0, 0, 255)       # red
}

# ── State ────────────────────────────────────────────────────────────────────
lanes           = {}
current_lane_id = 1
current_points  = []

def draw_state():
    img = clone.copy()

    # Draw all completed lanes
    for lid, pts in lanes.items():
        poly = np.array(pts, np.int32).reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], LANE_COLORS[lid])
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.polylines(img, [poly], True, LANE_COLORS[lid], 2)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(img, LANE_LABELS[lid], (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LANE_COLORS[lid], 2)

    # Draw current in-progress points
    for pt in current_points:
        cv2.circle(img, pt, 5, (255, 255, 0), -1)
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(img, current_points[i], current_points[i+1],
                     (255, 255, 0), 1)

    # Instructions at top
    if current_lane_id <= 4:
        msg = f"Drawing: {LANE_LABELS[current_lane_id]}  |  Points placed: {len(current_points)}"
    else:
        msg = "All 4 lanes done! Press S to save."

    cv2.rectangle(img, (0, 0), (640, 50), (0, 0, 0), -1)
    cv2.putText(img, msg, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(img,
                "LEFT click=place point | RIGHT click=finish lane | R=reset | S=save & quit",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    return img

def mouse_callback(event, x, y, flags, param):
    global current_points, current_lane_id

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_lane_id <= 4:
            current_points.append((x, y))
            print(f"  Point added: ({x}, {y})  — "
                  f"total points for Lane {current_lane_id}: {len(current_points)}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) >= 3 and current_lane_id <= 4:
            lanes[current_lane_id] = current_points.copy()
            print(f"Lane {current_lane_id} ({LANE_LABELS[current_lane_id]}) "
                  f"saved with {len(current_points)} points.")
            current_points = []
            current_lane_id += 1
            if current_lane_id <= 4:
                print(f"\nNow draw Lane {current_lane_id}: {LANE_LABELS[current_lane_id]}")
            else:
                print("\nAll 4 lanes drawn! Press S to save and quit.")
        elif len(current_points) < 3:
            print("Need at least 3 points to form a lane. Keep clicking.")

cv2.namedWindow("Lane Drawer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Lane Drawer", 900, 560)
cv2.setMouseCallback("Lane Drawer", mouse_callback)

print("=" * 55)
print("LANE DRAWING TOOL")
print("=" * 55)
print("Look at your road frame carefully.")
print("You will outline 4 lane zones as polygons.\n")
print("HOW TO:")
print("  - LEFT CLICK  to place corner points of the lane zone")
print("  - RIGHT CLICK to close and save the current lane")
print("  - R           to reset everything and start over")
print("  - S           to save and quit\n")
print("Start with Lane 1 (leftmost / slow lane)")
print("=" * 55)

while True:
    img = draw_state()
    cv2.imshow("Lane Drawer", img)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('r'):
        lanes.clear()
        current_lane_id = 1
        current_points  = []
        print("\nReset. Start over from Lane 1.")

    elif key == ord('s'):
        if len(lanes) < 4:
            print(f"Warning: Only {len(lanes)} lanes drawn. "
                  f"You need 4. Keep drawing or press S again to save partial.")
        os.makedirs("data", exist_ok=True)
        save_data = {
            "lanes":  {str(k): v for k, v in lanes.items()},
            "labels": LANE_LABELS
        }
        with open(OUTPUT_PATH, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nLane config saved to {OUTPUT_PATH}")
        print("You can now run the main pipeline.")
        break

    elif key == 27:   # Escape
        print("Cancelled.")
        break

cv2.destroyAllWindows()