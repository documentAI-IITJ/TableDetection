import os
import json
from glob import glob
from PIL import Image
from shutil import copyfile
from sklearn.model_selection import train_test_split
import fire

# Function to convert bounding box coordinates to YOLO format
def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Function to read JSON file and extract table annotations
def read_annotations(file_path, json_dir, images_dir):
    with open(os.path.join(json_dir, file_path)) as f:
        data = json.load(f)
    
    image_file = "_".join((file_path.split(".")[0]).split("_")[:-1]) + ".jpg"
    image_path = os.path.join(images_dir, image_file)
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    # Get annotations
    annotations_data = data.get("annotations", {})

    # YOLO format: class_id center_x center_y width height (all normalized)
    annotations = []

    for label, boxes in annotations_data.items():
        class_id = 0 if label == "HEADER" else 1  # Assign class IDs (0 for HEADER, 1 for FOOTER)
        
        for box in boxes:
            x_center = (box["X"] + box["W"] / 2) / img_width
            y_center = (box["Y"] + box["H"] / 2) / img_height
            width = box["W"] / img_width
            height = box["H"] / img_height
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return image_file, annotations

# Function to write YOLO format files
def write_yolo_files(split, annotations, images_dir, output_dir):
    for image_file, boxes in annotations.items():
        image_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(output_dir, 'labels', f"{image_name}.txt")
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, 'w') as f:
            for box in boxes:
                f.write(f"{box}\n")
        
        # Copy image to the corresponding directory
        src_image_path = os.path.join(images_dir, image_file)
        dst_image_path = os.path.join(output_dir, 'images', image_file)
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        copyfile(src_image_path, dst_image_path)

# Main script
def main(json_dir, images_dir, output_dir='data', test_size=0.2):
    """
    Converts JSON annotations to YOLO format, splits the dataset into training and testing sets, 
    and saves the resulting files in the specified directory structure.

    This function reads JSON annotation files from the given directory, converts the bounding box 
    annotations to YOLO format, and splits the dataset into training and testing sets based on 
    the specified test size. The converted annotations and associated images are then saved in 
    the appropriate directory structure for YOLO training.

    Args:
        json_dir (str): The directory containing the JSON annotation files.
        images_dir (str): The directory containing the corresponding image files.
        output_dir (str, optional): The base directory where the YOLO format files and images 
                                    will be saved. Defaults to 'data'.
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.

    Returns:
        None
    """

    all_annotations = {}
    for json_file in os.listdir(json_dir):
        image_file, annotations = read_annotations(json_file, json_dir, images_dir)
        all_annotations[image_file] = annotations
    
    # Split the dataset
    image_files = list(all_annotations.keys())
    annotations_list = [all_annotations[image_file] for image_file in image_files]
    
    train_files, test_files, train_annotations, test_annotations = train_test_split(
        image_files, annotations_list, test_size=test_size, random_state=42)
    
    train_annotations_dict = {file: ann for file, ann in zip(train_files, train_annotations)}
    test_annotations_dict = {file: ann for file, ann in zip(test_files, test_annotations)}
    
    # Create directories for YOLO dataset
    for split in ['train', 'test']:
        os.makedirs(f'{output_dir}/{split}/images', exist_ok=True)
        os.makedirs(f'{output_dir}/{split}/labels', exist_ok=True)
    
    # Write YOLO format files
    write_yolo_files('train', train_annotations_dict, images_dir, f'{output_dir}/train')
    write_yolo_files('test', test_annotations_dict, images_dir, f'{output_dir}/test')

if __name__ == "__main__":
    fire.Fire(main)