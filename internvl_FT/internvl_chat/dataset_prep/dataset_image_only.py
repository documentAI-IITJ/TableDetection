import os
import json
from PIL import Image

image_dir = "/scratch/data/tab2tex/tablev1.2/images/square/"
src_train_path = "/scratch/data/tab2tex/tablev1.2/structure/src-train.txt"
tgt_train_path = "/scratch/data/tab2tex/tablev1.2/structure/tgt-train.txt"
output_jsonl_path = "/scratch/data/tab2tex/train_dataset_ImageOnly_FAT_30k.jsonl"

with open(src_train_path, "r") as src_file:
    image_paths = [line.strip() for line in src_file.readlines()]

with open(tgt_train_path, "r") as tgt_file:
    ground_truths = [line.strip() for line in tgt_file.readlines()]

assert len(image_paths) == len(ground_truths), "Mismatch between src-train.txt and tgt-train.txt!"

base_prompt = """<image>

Given a table image, generate executable LaTeX code that accurately reconstructs the table's structure.

### **Guidelines:**
1. **Visual Information (from Image)**:
   - Detect grid structure (horizontal and vertical lines).
   - Identify row and column merging (e.g., spans across rows or columns).
   - Determine text alignment (left, center, right).
   - Ensure the correct number of rows and columns.
   - Treat all cell contents as **"CELL"** placeholders, regardless of actual text.

2. **Expected LaTeX Format**:
   - Start with `{ alignment }`Use appropriate column alignments (e.g., `{|l|c|r|}`, `{l l}`) based on the image.
   - Add `\\hline` for horizontal lines where visible.
   - Use `\\multirow` and `\\multicolumn` for merged cells.

**Expected Output**:
Generate valid executable LaTeX code that precisely represents the table.
"""

entries = []
total_images = len(image_paths)

for idx, (image_path, latex) in enumerate(zip(image_paths, ground_truths)):
 
    full_image_path = os.path.join(image_dir, image_path)

    if not os.path.exists(full_image_path):
        print(f"Warning: Image not found: {full_image_path}")
        continue

    with Image.open(full_image_path) as img:
        width, height = img.size

    entry = {
        "id": idx,
        "image": image_path, 
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

    if (idx + 1) % 1000 == 0 or idx + 1 == total_images:
        print(f"Processed {idx + 1}/{total_images} images.")

with open(output_jsonl_path, "w") as output_file:
    for entry in entries:
        output_file.write(json.dumps(entry) + "\n")

print(f"JSONL file created at: {output_jsonl_path}")