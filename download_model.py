from ultralytics import YOLO
import os

os.makedirs("models/weights", exist_ok=True)

print("Downloading YOLOv8 medium model...")
print("This will take a minute depending on your internet speed (~52MB)...")

model = YOLO("yolov8m.pt")

# Move the downloaded file to our models folder
import shutil
if os.path.exists("yolov8m.pt"):
    shutil.move("yolov8m.pt", "models/weights/yolov8m.pt")
    print("Model moved to models/weights/yolov8m.pt")
else:
    print("Model already saved by ultralytics cache.")

# Test the model loads correctly
model = YOLO("models/weights/yolov8m.pt")
print("\nModel loaded successfully!")
print(f"Number of classes: {len(model.names)}")
print("\nVehicle classes we will use:")
vehicle_classes = ["bicycle", "motorcycle", "car", "bus", "truck"]
for cls_id, name in model.names.items():
    if name in vehicle_classes:
        print(f"  Class ID {cls_id:3d} → {name}")