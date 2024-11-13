import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# Define category names and colors (e.g. 11 categories for DocLayNet)
categories = [
    {"id": 1, "name": "Caption", "color": "red"},
    {"id": 2, "name": "Footnote", "color": "blue"},
    {"id": 3, "name": "Formula", "color": "green"},
    {"id": 4, "name": "List-item", "color": "purple"},
    {"id": 5, "name": "Page-footer", "color": "orange"},
    {"id": 6, "name": "Page-header", "color": "pink"},
    {"id": 7, "name": "Picture", "color": "yellow"},
    {"id": 8, "name": "Section-header", "color": "brown"},
    {"id": 9, "name": "Table", "color": "cyan"},
    {"id": 10, "name": "Text", "color": "magenta"},
    {"id": 11, "name": "Title", "color": "lime"}
]
category_map = {cat['id']: cat for cat in categories}
def visualize_predictions_from_file(json_file_path, img_dir):
    # Load predictions from JSON file
    with open(json_file_path, "r") as f:
        predictions = json.load(f)
    os.makedirs(img_dir, exist_ok=True)
    def visualize_predictions(prediction):
        image_path = prediction["file_name"]
        instances = prediction["instances"]
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        image = Image.open(image_path)
        img_width, img_height = image.size
        # print(f"Image dimensions: {img_width}x{img_height}")
        # Define the original dimensions of the coordinate system
        original_width = 1025
        original_height = 1025
        # Compute scaling factors
        x_scale = img_width / original_width
        y_scale = img_height / original_height
        # print(f"Scaling factors: x_scale={x_scale}, y_scale={y_scale}")
        fig, ax = plt.subplots(1, figsize=(img_width / 100, img_height / 100), dpi=100)
        ax.imshow(image)
        for score, cls, bbox in zip(instances["scores"], instances["pred_classes"], instances["pred_boxes"]):
            color = category_map[cls + 1]["color"]
            label = category_map[cls + 1]["name"]
            bbox_scaled = [
                bbox[0] * x_scale, 
                bbox[1] * y_scale, 
                bbox[2] * x_scale, 
                bbox[3] * y_scale
            ]
            
            # print(f"Original bbox: {bbox}, Scaled bbox: {bbox_scaled}")
            rect = patches.Rectangle(
                (bbox_scaled[0], bbox_scaled[1]), 
                bbox_scaled[2] - bbox_scaled[0], 
                bbox_scaled[3] - bbox_scaled[1], 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(bbox_scaled[0], bbox_scaled[1] - 10, f'{label}: {score:.2f}', color=color, fontsize=12)
        ax.axis('off')
        output_path = os.path.join(img_dir, os.path.basename(image_path))
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    # Visualize predictions for all images in the predictions file
    for prediction in predictions:
        visualize_predictions(prediction)
json_file_path = "../../fts/single-test-image/inference/predictions.json"
img_dir = "../../fts/single-test-image/"
visualize_predictions_from_file(json_file_path, img_dir)