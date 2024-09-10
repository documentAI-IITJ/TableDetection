import os
import json
import fire
from PIL import Image

def convert_to_yolo_format(json_file, img_dir, output_dir):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Get the associated image file
    base_name = "_".join((os.path.splitext(os.path.basename(json_file))[0]).split("_")[:-1])
    image_file = os.path.join(img_dir, f"{base_name}.jpg")

    # Open image to get its dimensions
    with Image.open(image_file) as img:
        img_width, img_height = img.size

    # Get annotations
    annotations = data.get("annotations", {})

    # YOLO format: class_id center_x center_y width height (all normalized)
    yolo_data = []

    for label, boxes in annotations.items():
        class_id = 0 if label == "HEADER" else 1  # Assign class IDs (0 for HEADER, 1 for FOOTER)
        
        for box in boxes:
            x_center = (box["X"] + box["W"] / 2) / img_width
            y_center = (box["Y"] + box["H"] / 2) / img_height
            width = box["W"] / img_width
            height = box["H"] / img_height
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Prepare output file path
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    # Save to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(yolo_data))

def process_directory(json_dir, img_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each JSON file in the directory
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            convert_to_yolo_format(json_path, img_dir, output_dir)

def main(json_dir, img_dir, output_dir):
    """
    Processes a directory of JSON annotation files and converts each to YOLO format.

    This function iterates through all JSON files in the specified directory, reads the corresponding image file
    to determine its dimensions, and converts the annotations in each JSON file to the YOLO format. The resulting
    YOLO annotations are saved as .txt files with the same base name as the JSON files in the specified output directory.

    Args:
        json_dir (str): The directory containing the JSON annotation files.
        img_dir (str): The directory containing the image files associated with the JSON annotations.
        output_dir (str): The directory where the YOLO format .txt files will be saved.

    Returns:
        None
    """
    process_directory(json_dir, img_dir, output_dir)

if __name__ == "__main__":
    fire.Fire(main)
