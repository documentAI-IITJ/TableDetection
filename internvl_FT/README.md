# Table2Tex with Fine-Tuned VLM (InternVL2.5_4B)

This repository contains all scripts, configuration files, and resources used for my M.Tech thesis project: **"Table2Tex: Empirical Study of VLM Training Regimes"**.

We fine-tune [InternVL2.5](https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html) on the Tab2Tex dataset to convert document table images into structured LaTeX code using various training strategies.

---

## 🗂️ Directory Structure
```
internvl_FT/
├── shell/                        # Shell scripts for environment and GPU setup
│   ├── zero_stage1_config.json
│   └── allocate_GPU.sh
├── dataset_prep/                        # Instructions and paths to download Tab2Tex
│   └── dataset_prep.py
├── environment.yml              # internvl Conda environment definition
├── notebooks/                   # Jupyter notebooks for training & analysis
│   ├── finetune_inference.ipynb
├── results/                     # Output predictions, logs, metrics
│   └── summary_results.txt
├── README.md                    # This file
└── codeBLEU_detailed_Eval_metrics.py  # Eval Script
└── EA_overall_summary.py             # Eval Script
```

---

## ⚙️ Environment Setup

Use the provided `environment.yml` file to create your environment:
```bash
conda env create -f environment.yml
conda activate internvl
```

---

## 🚀 Finetuning InternVL2.5

Follow [InternVL's official finetuning instructions](https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html).

### 🔹 Step 1: Download Pretrained Weights
```bash
# Example path to download OpenGVLab weights
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-4B --local-dir InternVL2_5-4B
```

### 🔹 Step 2: Prepare the Dataset
Download the [Tab2Tex dataset](https://drive.google.com/drive/folders/19nRoEpVJlIYtVTNQ5Kcit2HTCgfqf4mj) and place it as:
```
data/
└── tab2tex/
    ├── images/
```

### 🔹 Step 3: Run Finetuning
```bash
bash bash/finetune_internvl.sh
```
This uses 2 A100 GPUs (40GB) and follows configs from `configs/internvl_finetune.yaml`.

---

## 🔍 Inference
Run predictions using your fine-tuned model checkpoint:
```bash
python scripts/inference.py --config configs/internvl_finetune.yaml
```

---

## 📊 Evaluation
Summarize results to get metrics like EA, E95, and average similarity:
```bash
python scripts/evaluate.py \
```

---

## 📓 Notebooks
| Notebook                       | Description                                 |
|-------------------------------|---------------------------------------------|
| `finetune_row_col_latex.ipynb` | Stagewise training on row/col → LaTeX       |
| `inference_internvl.ipynb`     | Predict and visualize LaTeX outputs         |
| `failure_analysis.ipynb`       | Analyze prediction mismatches/errors        |

---

## 📌 Notes
- Pretrained weights are hosted by OpenGVLab via HuggingFace.
- Shell scripts assume SLURM-based GPU environment but can be adapted.
- Evaluation script outputs per-sample metrics and aggregates.

---

## 📎 Related Links
- Official Finetuning Guide: [InternVL Docs](https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html)
- Dataset: [Tab2Tex Google Drive](https://drive.google.com/drive/folders/19nRoEpVJlIYtVTNQ5Kcit2HTCgfqf4mj)
- Project Directory: [documentAI-IITJ/TableDetection](https://github.com/documentAI-IITJ/TableDetection/tree/tab2tex-vlm/internvl_FT)

---

## 🏁 License
MIT License

---

## ✍️ Author
**Ritu Singh (M23CSE017)**
Email: m23cse017@iitj.ac.in
