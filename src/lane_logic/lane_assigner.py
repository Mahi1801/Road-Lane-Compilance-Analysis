import json
import numpy as np
import cv2

ALLOWED_CATEGORIES = {
    1: ["cyclist"],
    2: ["bike"],
    3: ["car"],
    4: ["truck"]
}

class LaneAssigner:
    def __init__(self, config_path="data/lane_config.json"):
        with open(config_path) as f:
            config = json.load(f)
        self.lanes = {
            int(k): np.array(v, np.int32)
            for k, v in config["lanes"].items()
        }
        self.labels = config["labels"]
        print(f"Lane assigner ready. Loaded {len(self.lanes)} lane zones.")

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    def get_lane(self, point):
        for lane_id, polygon in self.lanes.items():
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return lane_id
        return None

    def is_violation(self, category, lane_id):
        if lane_id is None:
            return False
        allowed = ALLOWED_CATEGORIES.get(lane_id, [])
        return category not in allowed

    def assign(self, tracked_vehicles):
        enriched = []
        for vehicle in tracked_vehicles:
            centroid  = self.get_centroid(vehicle["bbox"])
            lane_id   = self.get_lane(centroid)
            violation = self.is_violation(vehicle["category"], lane_id)
            enriched.append({
                **vehicle,
                "lane_id":  lane_id,
                "centroid": centroid,
                "violation": violation
            })
        return enriched

    def draw_lanes(self, frame, alpha=0.15):
        overlay = frame.copy()
        COLORS  = {1:(0,255,0), 2:(0,165,255), 3:(255,100,0), 4:(0,0,255)}
        for lid, poly in self.lanes.items():
            cv2.fillPoly(overlay, [poly], COLORS.get(lid, (128,128,128)))
            cv2.polylines(overlay, [poly], True, COLORS.get(lid,(128,128,128)), 2)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)