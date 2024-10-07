import fire
from ultralytics import YOLO
import numpy as np
import cv2
import os
import json

def create_label_mapping():
    """Define a mapping for label classes."""
    label_mapping = {
        0: "Table",
    }
    return label_mapping

def convert_to_json(filename, cls_list, conf_list, xyxy_list):
    """Convert prediction data to JSON format.

    Args:
        filename (str): Name of the image file.
        cls_list (list): List of class indices.
        conf_list (list): List of confidence scores.
        xyxy_list (list): List of bounding box coordinates in xyxy format.

    Returns:
        str: JSON-formatted string of the prediction data.
    """
    label_map = create_label_mapping()
    json_data = {
        "filename" : filename,
        "detections": [
            {
                "cls": int(cls_list[i]),
                "cls_name" : label_map.get(int(cls_list[i]), "Unknown label"),
                "conf": float(conf_list[i]),
                "bbox": 
                {
                    "x1": float(xyxy_list[i][0]),
                    "y1": float(xyxy_list[i][1]),
                    "x2": float(xyxy_list[i][2]),
                    "y2": float(xyxy_list[i][3])
                },
            }
            for i in range(len(cls_list))
        ]
    }
    return json.dumps(json_data, indent=4)

def save_prediction(image_dir, model_path, save_dir):
    """Run model prediction on images and save the results.

    Args:
        image_dir (str): Directory containing images to be processed.
        model_path (str): Path to the YOLO model weights file.
        save_dir (str): Directory to save images with predictions drawn.
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = YOLO(model_path)
    for image in os.listdir(image_dir):
        frame = cv2.imread(os.path.join(image_dir, image))
        results = model.predict(frame, save=False)

        json_string = convert_to_json(image, results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xyxy)

        filename = image.split(".")[0]
        os.makedirs("predictions/jsons", exist_ok=True)
        with open(f"predictions/jsons/{filename}.json", 'w') as file:
            file.write(json_string)

        img = results[0].plot(font_size=20, pil=True)
        cv2.imwrite(f"{save_dir}/{image}", img)

if __name__ == "__main__":
    fire.Fire(save_prediction)


# python your_script.py --image_dir "/path/to/images" --model_path "model/best.pt" --save_dir "/path/to/save_dir"
