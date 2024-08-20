from ultralytics import YOLO
import numpy as np
import cv2
import os


model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)
# Customize validation settings
validation_results = model.val(data='doclaynet_large.yaml',
                                save_json=True,
                                device=[0],
                                batch=100,
                                conf=0.3,
                                iou=0.7)

# print(validation_results)
