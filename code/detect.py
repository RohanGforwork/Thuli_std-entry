from ultralytics import YOLO

model = YOLO("trained_model/best.pt")
model.predict(source="input/bengali.avi", conf=0.25, save=True)

