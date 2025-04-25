from ultralytics import YOLO

model = YOLO("best.pt")
model.predict(source="bengali.mp4", conf=0.25, save=True)

