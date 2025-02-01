from ultralytics import YOLO


model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

results = model.train(data="widerface-yolo/dataset.yaml", epochs=200, imgsz=640)
