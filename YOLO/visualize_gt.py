import os
import json
import cv2
import matplotlib.pyplot as plt

class_mapping = {
    0: "Header",
    1: "Footer"
}

def read_annotations(filename):
    """
    Read annotations from YOLO format text file.
    Args:
    - filename: Path to the YOLO format text file.

    Returns:
    - annotations: List of annotations in the format [(class, x_center, y_center, width, height), ...].
    """
    annotations = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append((class_id, x_center, y_center, width, height))
    return annotations


def plot_annotations(image_filename, annotations):
    """
    Plot annotations on the image using OpenCV.
    Args:
    - image_filename: Path to the image file.
    - annotations: List of annotations in the format [(class, x_center, y_center, width, height), ...].
    """
    image = cv2.imread(image_filename)
    image_height, image_width, _ = image.shape

    for annotation in annotations:
        class_id, x_center, y_center, width, height = annotation

        # Convert YOLO format to absolute coordinates
        x1 = int((x_center - width/2) * image_width)
        y1 = int((y_center - height/2) * image_height)
        x2 = int((x_center + width/2) * image_width)
        y2 = int((y_center + height/2) * image_height)

        # Plot rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Plot text
        class_label = f"{class_mapping[class_id]}"
        cv2.putText(image, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("check.png", image)

if __name__ == "__main__":

    filename = "<filename>.png"
    folder = "test"

    image_filename = f"data/{folder}/images/{filename}.jpg"  # Change this to your image filename
    annotation_filename = f"data/{folder}/labels/{filename}.txt"  # Change this to your annotations filename

    annotations = read_annotations(annotation_filename)
    plot_annotations(image_filename, annotations)
