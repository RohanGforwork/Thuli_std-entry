from ultralytics import YOLO

model = YOLO("trained_model/dress.pt")
model .predict(source="input/bengali.mp4", conf=0.25, save=True)

