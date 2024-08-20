import os
import json
from glob import glob
from PIL import Image
from shutil import copyfile
from sklearn.model_selection import train_test_split

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
def read_annotations(file_path, images_dir):
    with open(file_path) as f:
        data = json.load(f)
    
    image_file = data['file_name']
    image_path = os.path.join(images_dir, image_file)
    with Image.open(image_path) as img:
        image_width, image_height = img.size
    
    annotations = []
    for table in data['tables']:
        coords = table['coordinates']
        box = (coords['x1'], coords['x2'], coords['y1'], coords['y2'])
        yolo_box = convert_to_yolo((image_width, image_height), box)
        annotations.append(yolo_box)
    
    return image_file, annotations

# Function to write YOLO format files
def write_yolo_files(split, annotations, images_dir, output_dir):
    for image_file, boxes in annotations.items():
        image_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(output_dir, 'labels', f"{image_name}.txt")
        with open(label_file, 'w') as f:
            for box in boxes:
                f.write(f"0 {' '.join(map(str, box))}\n")
        
        # Copy image to the corresponding directory
        src_image_path = os.path.join(images_dir, image_file)
        dst_image_path = os.path.join(output_dir, 'images', image_file)
        copyfile(src_image_path, dst_image_path)

# Main script
def main():
    json_files = glob('/DATA/mishra/table_work/datasets/whole_dataset/jsons/*.json')  # Adjust the path to your JSON files
    images_dir = '/DATA/mishra/table_work/datasets/whole_dataset/images'  # Adjust the path to your images directory
    
    all_annotations = {}
    for json_file in json_files:
        image_file, annotations = read_annotations(json_file, images_dir)
        all_annotations[image_file] = annotations
    
    # Split the dataset
    image_files = list(all_annotations.keys())
    annotations_list = [all_annotations[image_file] for image_file in image_files]
    
    train_files, test_files, train_annotations, test_annotations = train_test_split(
        image_files, annotations_list, test_size=0.2, random_state=42)
    
    train_annotations_dict = {file: ann for file, ann in zip(train_files, train_annotations)}
    test_annotations_dict = {file: ann for file, ann in zip(test_files, test_annotations)}
    
    # Create directories for YOLO dataset
    for split in ['train', 'test']:
        os.makedirs(f'/DATA/himanshi1/datasets/rapidAI_yolo/{split}/images', exist_ok=True)
        os.makedirs(f'/DATA/himanshi1/datasets/rapidAI_yolo/{split}/labels', exist_ok=True)
    
    # Write YOLO format files
    write_yolo_files('train', train_annotations_dict, images_dir, '/DATA/himanshi1/datasets/rapidAI_yolo/train')
    write_yolo_files('test', test_annotations_dict, images_dir, '/DATA/himanshi1/datasets/rapidAI_yolo/test')

if __name__ == "__main__":
    main()
