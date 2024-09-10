#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6 python -m torch.distributed.launch \
  --nproc_per_node=5 \
  --nnodes=1 \
  --node_rank=0 \
  --use_env \
  main.py \
  --batch_size 16 \
  --epochs 50 \
  --coco_path /DATA/dataset/DocLayNet/DocLayNet_core/doclaynet_detr \
  --num_workers 32 \
  --resume pretrained/detr-r50-e632da11.pth \
  --output_dir outputs
