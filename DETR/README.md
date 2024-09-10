# Train Detr on Custom Dataset

## Setup
```
conda create -n detr_env python=3.7
conda activate detr_env

# for CUDA 10.2
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

pip install pycocotools
```
This is enough to get you running.

## Data Preparation
Keeping the data in the following structure will let you use the ```datasets/coco.py``` that has been modified.
```
doclaynet_detr/
├── test
│   ├── annotation.json -> ../../COCO/test.json
│   └── images -> ../../PNG_test/
├── train
│   ├── annotation.json -> ../../COCO/train.json
│   └── images -> ../../PNG_train/
└── val
    ├── annotation.json -> ../../COCO/val.json
    └── images -> ../../PNG_val/
```
Don't worry about these "->". These are just symbolic links so as to not have multiple copies of the same thing.
Line 117-181 shows the structure that has been used.

```
    PATHS = {
        "train": (root / "train/images", root / "train" / f'annotation.json'),
        "val": (root / "val/images", root / "val" / f'annotation.json'),
        "test": (root / "test/images", root / "test" / f'annotation.json'),
    }

```
comma separates the image folder with the annotation file.

## Training
```
chmod +x train.sh
./train.sh
```
Modifications are to be done accordingly for datapath, batch_size etc.  If you are using single GPU the edit the train.sh file in the following manner.
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --use_env \
....
```
The pretrained DETR model with backbone R50, schedule 500, inf_time 0.036 and boxAP 42.0 of size 159Mb can be downloaded from [this link](https://drive.google.com/file/d/1V0p4_hQKPPEpmTJ1YYnK9BVQPdWKLzxk/view?usp=sharing).

## Evaluation
Please to the modificaitons as needed.
```
chmod +x eval.sh
./eval.sh
```
