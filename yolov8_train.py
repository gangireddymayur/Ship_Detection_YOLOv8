from ultralytics import YOLO

def train_yolov8():
    model = YOLO("yolov8n.pt")

    model.train(
        data="D:/Ship_Detection_YOLOv8/ships-aerial-images/data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        project="shipdetection",
        name="yolov8_ship",
        patience=5,
        pretrained=True
    )

if __name__ == "__main__":
    train_yolov8()
