from ultralytics import YOLO
import numpy as np
import cv2
import os
import json

def create_label_mapping():
    # Define the mapping in a dictionary where the keys are integers and the values are the corresponding descriptions
    label_mapping = {
        0: "Table",
    }
    return label_mapping

def convert_to_json(filename, cls_list, conf_list, xyxy_list):
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
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = YOLO(model_path)
    count = 0
    # model.predict(image_dir, save=True, imgsz=640, conf=0.7)
    for image in os.listdir(image_dir):
        frame = cv2.imread(os.path.join(image_dir, image))
        results = model.predict(frame, save=False)

        json_string = convert_to_json(image, results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xyxy)

        filename = image.split(".")[0]
        os.makedirs("predictions_rapidai/jsons", exist_ok=True)
        with open(f"predictions_rapidai/jsons/{filename}.json", 'w') as file:
            file.write(json_string)

        img = results[0].plot(font_size=20, pil=True)
        cv2.imwrite(f"{save_dir}/{image}", img)
        # if count == 5: break
        # count += 1


if __name__ == "__main__":

    image_dir = "/DATA/himanshi1/datasets/rapidAI_yolo/test/images" 
    model_path = "runs/detect/train/weights/best.pt"
    save_dir = "predictions_rapidai/images"

    
    save_prediction(image_dir, model_path, save_dir)