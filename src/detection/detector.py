from ultralytics import YOLO
import cv2

VEHICLE_CATEGORY_MAP = {
    "bicycle":    "cyclist",
    "motorcycle": "bike",
    "car":        "car",
    "bus":        "car",
    "truck":      "truck",
}

ALLOWED_CLASSES = list(VEHICLE_CATEGORY_MAP.keys())

class VehicleDetector:
    def __init__(self, model_path="models/weights/yolov8m.pt", confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.allowed_class_ids = [
            cls_id for cls_id, name in self.model.names.items()
            if name in ALLOWED_CLASSES
        ]
        print(f"Detector ready. Watching for: {ALLOWED_CLASSES}")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.allowed_class_ids,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf     = float(box.conf[0])
            category = VEHICLE_CATEGORY_MAP.get(cls_name, "unknown")

            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "class_name": cls_name,
                "category":   category,
                "confidence": round(conf, 3)
            })

        return detections

    def draw_detections(self, frame, detections):
        COLOR_MAP = {
            "cyclist": (0, 255, 0),
            "bike":    (0, 165, 255),
            "car":     (255, 100, 0),
            "truck":   (0, 0, 255),
            "unknown": (128, 128, 128)
        }
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = COLOR_MAP.get(d["category"], (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['category']} {d['confidence']}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame