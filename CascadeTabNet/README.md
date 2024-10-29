# table-detection
## Installation 
As recommended from mmdetection, [prerequisites](https://mmdetection.readthedocs.io/en/stable/get_started.html#prerequisites), [installation](https://mmdetection.readthedocs.io/en/stable/get_started.html#installation)

## Training
Navigate to the model config file
```
./table-detection/mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py
```
Locate the dataset config file
```
./table-detection/mmdetection/configs/_base_/datasets/coco_detection.py
```
Here set the coco_datapath, and batchsize as required.
Next, we need to mention about classes and color palletes. In the mmdetection, coco_dataset config
```
./table-detection/mmdetection/mmdet/datasets/coco.py
``` 
And then for the preferred model cofig, change the num_classes
```
./table-detection/mmdetection/configs/_base_/models/cascade-rcnn_r50_fpn.py
```

Training on a single GPU run
```
cd mmdetection
python tools/train.py configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py
```

Training on multiple GPUs
```
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```
```
cd mmdetection
./tools/dist_train.sh configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py 2 
```

## FAQs
The following [github issue](https://github.com/open-mmlab/mmdetection/issues/9610) solved the problem
```
  File "<__array_function__ internals>", line 200, in concatenate
ValueError: need at least one array to concatenate
``` 
These [documentations](https://mmdetection.readthedocs.io/en/stable/user_guides/index.html) helped a lot to understand the training, testing, and data preparation.

# table-detection
