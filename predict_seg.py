from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load a model
model = YOLO("runs/segment/train13/weights/best.pt")  # load a custom model

img_path = "../CircleLineDetector/res/1.bmp"
# Predict with the model
results = model(img_path)  # predict on an image
img = cv2.imread(img_path)

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    # Create matplotlib figure
    plotted_image = result.plot()
    plt.figure(figsize=(12, 8))
    plt.imshow(plotted_image)
    
    plt.show()