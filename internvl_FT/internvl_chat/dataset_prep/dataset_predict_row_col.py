import os
import json
from PIL import Image

image_dir = "/scratch/data/tab2tex/tablev1.2/images/square/"
src_train_path = "/scratch/data/tab2tex/tablev1.2/structure/src-test.txt"
tgt_train_path = "/scratch/data/tab2tex/table_counts_test_cell_based.txt"
output_jsonl_path = "/scratch/data/tab2tex/test_dataset_ImageOnly_FAT_row_col.jsonl"

with open(src_train_path, "r") as src_file:
    image_paths = [line.strip() for line in src_file.readlines()]

with open(tgt_train_path, "r") as tgt_file:
    ground_truths = [line.strip() for line in tgt_file.readlines()]

assert len(image_paths) == len(ground_truths), "Mismatch between src-train.txt and tgt-train.txt!"

base_prompt = """<image>
You are an expert table analysis model. Given an image of a table, your task is to determine the number of rows and columns based on the visible structure in the image. Count only the actual rows and columns as they appear, including any merged cells, and ignore any empty or hidden cells. Provide your answer in the format: 'rows: X, columns: Y', where X is the number of rows and Y is the number of columns. Do not include any additional text or explanation.
"""

# Generate JSONL entries
entries = []
total_images = len(image_paths)

for idx, (image_path, latex) in enumerate(zip(image_paths, ground_truths)):
    # Construct full image path
    full_image_path = os.path.join(image_dir, image_path)

    # Verify the image exists
    if not os.path.exists(full_image_path):
        print(f"Warning: Image not found: {full_image_path}")
        continue

    # Get image dimensions
    with Image.open(full_image_path) as img:
        width, height = img.size

    # Create JSONL entry
    entry = {
        "id": idx,
        "image": image_path,  # Relative path to the image
        "width": width,
        "height": height,
        "conversations": [
            {
                "from": "human",
                "value": base_prompt
            },
            {
                "from": "gpt",
                "value": latex
            }
        ]
    }
    entries.append(entry)

    # Print progress every 1000 images
    if (idx + 1) % 1000 == 0 or idx + 1 == total_images:
        print(f"Processed {idx + 1}/{total_images} images.")

# Write JSONL file
with open(output_jsonl_path, "w") as output_file:
    for entry in entries:
        output_file.write(json.dumps(entry) + "\n")

print(f"JSONL file created at: {output_jsonl_path}")
