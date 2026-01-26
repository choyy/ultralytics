from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load a model
model = YOLO("runs/detect/train9/weights/best.pt")  # load a custom model

img_path = "../CircleLineDetector/res/1.bmp"
# Predict with the model
results = model(img_path)  # predict on an image

# Plot the results
for result in results:
    plotted_image = result.plot()  # Use built-in plot method
    
    # Convert BGR to RGB for matplotlib
    plotted_image = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
    
    # Create matplotlib figure
    plt.figure(figsize=(12, 8))
    plt.imshow(plotted_image)
    plt.title('Detection Results')
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.show()