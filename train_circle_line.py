from ultralytics import YOLO
def main():
    data = "ultralytics/cfg/datasets/ellipse-line-obb.yaml"

    # Load a model
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data, epochs=50, batch=32,
                        imgsz=640, device="0")
if __name__ == '__main__':
    main()