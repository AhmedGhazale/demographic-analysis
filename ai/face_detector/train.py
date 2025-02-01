from ultralytics import YOLO



if __name__ == "__main__":

    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="widerface-yolo/dataset.yaml", epochs=200, imgsz=640)
