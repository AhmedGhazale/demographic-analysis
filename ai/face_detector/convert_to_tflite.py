from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train2/weights/best.pt")  # load an official model


# Export the model
model.export(format="tflite")