import os
import json
from PIL import Image

image_dir = "/scratch/data/tab2tex/tablev1.2/images/square/" 
src_path = "/scratch/data/tab2tex/tablev1.2/structure/src-train.txt"  
tgt_path = "/scratch/data/tab2tex/tablev1.2/structure/tgt-train.txt"
row_col_path = "/scratch/data/tab2tex/table_counts_train_cell_based.txt" 
output_jsonl_path = "/scratch/data/tab2tex/train_dataset_combined_row_col_latex.jsonl"

# Read image paths, LaTeX ground truths, and row/column counts
with open(src_path, "r") as src_file:
    image_paths = [line.strip() for line in src_file.readlines()]

with open(tgt_path, "r") as tgt_file:
    latex_truths = [line.strip() for line in tgt_file.readlines()]

with open(row_col_path, "r") as rc_file:
    row_col_truths = [line.strip() for line in rc_file.readlines()]

assert len(image_paths) == len(latex_truths) == len(row_col_truths), "Mismatch between input files!"

combined_prompt = """<image>
You are an expert table analysis model. Given an image of a table, your task is to:

1. Count the number of rows and columns in the table structure.
2. Generate executable LaTeX code that accurately reconstructs the table's structure.

### Part 1: Row and Column Count
Determine the number of rows and columns based on the visible structure in the image. 
Count only the actual rows and columns as they appear, including any merged cells.
Format: "rows: X, columns: Y"

### Part 2: LaTeX Structure
Generate valid LaTeX code that precisely represents the table structure with the following guidelines:
- Start with the column alignment specification (e.g., `{|l|c|r|}`)
- Add `\\hline` for horizontal lines where visible
- Use `\\multirow` and `\\multicolumn` for merged cells
- Replace all text content with **"CELL"** placeholders

Provide your complete answer in this format:
```
Row and Column Count:
rows: X, columns: Y

LaTeX Structure:
[Your LaTeX code here]
```
"""


entries = []
total_images = len(image_paths)

for idx, (image_path, latex_truth, row_col_truth) in enumerate(zip(image_paths, latex_truths, row_col_truths)):
    full_image_path = os.path.join(image_dir, image_path)

    if not os.path.exists(full_image_path):
        print(f"Warning: Image not found: {full_image_path}")
        continue

    with Image.open(full_image_path) as img:
        width, height = img.size


    combined_truth = f"""Row and Column Count:
{row_col_truth}

LaTeX Structure:
{latex_truth}"""

    entry = {
        "id": idx,
        "image": image_path,  # Relative path to the image
        "width": width,
        "height": height,
        "conversations": [
            {
                "from": "human",
                "value": combined_prompt
            },
            {
                "from": "gpt",
                "value": combined_truth
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