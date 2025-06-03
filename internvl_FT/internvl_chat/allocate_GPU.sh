#!/bin/bash
#SBATCH --job-name=jupyter_job_fat
#SBATCH --output=jupyter_job_%j.out
#SBATCH --error=jupyter_job_%j.err
#SBATCH --partition=dgx  # Replace with your partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 ##Define number of GPUs
date;hostname;pwd

module load anaconda3/2024

source activate 

conda activate internvl

# Start the Jupyter server
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root