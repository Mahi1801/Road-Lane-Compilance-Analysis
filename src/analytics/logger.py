import csv
import os

class TrafficLogger:
    def __init__(self, output_path="outputs/logs/traffic_log.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output_path = output_path
        self.fieldnames  = [
            "frame_id", "track_id", "category",
            "lane_id",  "violation",
            "centroid_x", "centroid_y",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        print(f"Logger ready. Writing to {output_path}")

    def log(self, frame_id, vehicles):
        rows = []
        for v in vehicles:
            cx, cy   = v.get("centroid", (0, 0))
            x1,y1,x2,y2 = v["bbox"]
            rows.append({
                "frame_id":   frame_id,
                "track_id":   v["track_id"],
                "category":   v["category"],
                "lane_id":    v.get("lane_id"),
                "violation":  v.get("violation", False),
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox_x1": x1, "bbox_y1": y1,
                "bbox_x2": x2, "bbox_y2": y2,
            })
        if rows:
            with open(self.output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerows(rows)