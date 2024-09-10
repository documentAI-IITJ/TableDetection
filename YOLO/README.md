# YOLO RapidAI HeaderFooter Detection

This repository contains scripts and resources for converting annotation data from JSON format to YOLO format, training a YOLO model for header and footer detection, and visualizing the ground truth annotations.

## Directory Structure

```
yolo_rapidAI_headerFooter/
├── check.png                 # Sample image for checking
├── convert_data_to_yolo.py   # Script for converting JSON to YOLO format
├── data/                     # Directory for storing datasets
├── main.py                   # Main script for data conversion and processing
├── rapidAI_data_to_yolo.py   # Alternate script for data conversion and data splitting
├── runs/                     # Directory for storing training results
├── train.yaml                # YAML configuration file for training
├── visualize_gt.py           # Script for visualizing ground truth annotations
└── yolov8n.pt                # Pre-trained YOLOv8n model
```

## Installation

Create a virtual environment with python >= 3.9 and install the following python packages.
```
pip install opencv-python
pip install pillow
pip install matplotlib
pip install fire 
```

## Converting Data from JSON to YOLO Format

Use the ```convert_data_to_yolo.py``` script to convert your JSON annotations to YOLO format. This script processes a directory of JSON files and saves the converted annotations in the specified output directory.

Usage
```
python convert_data_to_yolo.py json_dir images_dir output_dir
```
Also use ``` python convert_data_to_yolo.py --help ``` to know more.

Use the ``` rapidAI_data_to_yolo.py``` script which converts JSON annotations to YOLO format, splits the dataset into training and testing sets, and saves the resulting files in the specified directory structure.

Usage
```
python rapidAI_data_to_yolo.py json_dir images_dir
```

Also use ``` python rapidAI_data_to_yolo.py --help ``` to know more.


## Training the YOLO Model
To train the YOLO model using the converted data, use the ```main.py``` script.
Usage
```
python main.py --yaml_file train.yaml --epochs 20 --devices [1] --batch 128
```
Also use ``` python main.py --help ``` to know more.


## Visualizing Ground Truth Annotations
The ```visualize_gt.py``` script allows you to visualize the ground truth annotations on the images. This create a png file named as "check.png" by default. 