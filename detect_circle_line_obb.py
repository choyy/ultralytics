from ultralytics import YOLO
def main():
    data = "ultralytics/cfg/datasets/ellipse-line-obb.yaml"

    # Load a model
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data, epochs=100, batch=32,
                        retina_masks=True, close_mosaic=0, \
                        optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, \
                        translate=0.0, momentum=0.9, \
                        overlap_mask=True, scale=0.5, mixup=0.0, copy_paste=0.0,\
                        imgsz=640, device="0")
if __name__ == '__main__':
    main()