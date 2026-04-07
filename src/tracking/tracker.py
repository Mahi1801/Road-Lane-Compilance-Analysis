from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame):
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        ds_input = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], d["confidence"], d["category"]))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        tracked = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            tracked.append({
                "track_id":  track.track_id,
                "bbox":      [x1, y1, x2, y2],
                "category":  track.det_class,
                "confidence": track.det_conf or 0.0
            })

        return tracked