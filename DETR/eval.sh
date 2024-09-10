#!/bin/bash
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --use_env \
    eval.py \
 --coco_path /DATA/dataset/DocLayNet/DocLayNet_core/doclaynet_detr \
 --resume outputs/checkpoint.pth \
 --batch_size 2 \
 --eval 