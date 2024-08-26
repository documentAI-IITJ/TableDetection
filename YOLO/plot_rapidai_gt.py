import json
import cv2
import os
from tqdm import tqdm
import shutil



def copy_item(src, dst):
    shutil.copy(src, dst)

def plot_boxes(image_path, json_path, output_dir):
    # Read the JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    print(data["annotations"])

    # # Read the image
    image = cv2.imread(image_path)
    
    # # Plot bounding boxes for tables
    for label, boxes in data["annotations"].items():
            for box in boxes:
                x, y, w, h = box['X'], box['Y'], box['W'], box['H']
                # Draw rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put text above the bounding box
                cv2.putText(image, label, ((x+w) - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    image_filename = image_path.split("/")[-1]
    output_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving


image_dir = "/DATA/himanshi1/datasets/Header_footer_dataset/images"
json_dir = "/DATA/himanshi1/datasets/Header_footer_dataset/jsons"

# image_dir_new = "/DATA/himanshi1/datasets/Header_footer_dataset/images"
# json_dir_new = "/DATA/himanshi1/datasets/Header_footer_dataset/jsons"

# os.makedirs(image_dir_new, exist_ok=True)
# os.makedirs(json_dir_new, exist_ok=True)

output_dir = "/DATA/himanshi1/scripts/yolo_HeaderFooter/predictions_rapidai/gt_on_preds"

os.makedirs(output_dir, exist_ok=True)
count = 0
for file in tqdm(os.listdir(image_dir)):
    
    image_file = os.path.join(image_dir, file)

    # print(image_file)
    json_file = os.path.join(json_dir, file.split(".")[0]+"_output"+".json")
    # # print(json_file)
    # # break

    # if os.path.exists(json_file) == False:
    #     count += 1
    #     print(f"image:{image_file}\njson:{json_file}\ncount:{count}")

    # # elif os.path.exists(json_file) == True and os.path.exists(image_file)==True:
    # #     copy_item(image_file, image_dir_new)
    # #     copy_item(json_file, json_dir_new)

    plot_boxes(image_file, json_file, output_dir)
    # break