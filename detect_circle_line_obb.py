from ultralytics import YOLO
def main():
    data = "ultralytics/cfg/datasets/ellipse-line-obb.yaml"

    # Load a model
    model = YOLO("yolov8n-obb.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data, epochs=50, batch=64,
                        retina_masks=True, mosaic=0.0 , close_mosaic=0,
                        # use_kld=True,  # https://github.com/ultralytics/ultralytics/pull/16851#issuecomment-2662315383
                        optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0,
                        translate=0.0, momentum=0.9,
                        shear=0.0, perspective=0.0, # 不使用剪切和透视
                        overlap_mask=True, scale=0.5, mixup=0.0, copy_paste=0.0,
                        imgsz=640, device="0")
if __name__ == '__main__':
    main()