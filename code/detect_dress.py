from ultralytics import YOLO

model = YOLO("dress.pt")
model .predict(source="bengali_detected.avi", conf=0.25, save=True)

