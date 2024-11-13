#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------
"""
Detection Inference Script for MPViT.
"""
import os
import torch
import json
import cv2
from typing import Any, Dict, List, Set
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import inference_on_dataset
from detectron2.solver.build import maybe_add_gradient_clipping
from ditod import add_vit_config
from ditod import DetrDatasetMapper
from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import MyDetectionCheckpointer, ICDAREvaluator
from ditod import MyTrainer
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  # Freeze the configuration to prevent further modifications
    default_setup(cfg, args)
    return cfg
def run_inference(cfg, model):
    """
    Run inference on the test dataset.
    """
    # Ensure that the model is set for inference
    model.eval()  # Set model to evaluation mode
    test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    predictions = []
    for inputs in test_loader:
        with torch.no_grad():
            outputs = model(inputs)
        for input, output in zip(inputs, outputs):
            instances = output["instances"].to("cpu").get_fields()
            prediction = {
                "image_id": input["image_id"],
                "file_name": input["file_name"],
                "instances": {
                    "scores": instances["scores"].tolist(),
                    "pred_classes": instances["pred_classes"].tolist(),
                    "pred_boxes": [box.tolist() for box in instances["pred_boxes"].tensor]
                }
            }
            predictions.append(prediction)
    # Save predictions to a JSON file in the OUTPUT_DIR
    inference_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    output_file = os.path.join(inference_dir, "predictions.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f)
    # Log completion message
    logger = logging.getLogger("detectron2")
    logger.info("Inference completed. Predictions saved to %s", output_file)
    return predictions
def main(args):
    cfg = setup(args)
    """
    Register test dataset without annotations
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.structures import BoxMode
    def get_test_dicts(img_dir):
        dataset_dicts = []
        for idx, filename in enumerate([file for file in os.listdir(img_dir) if file.endswith(".jpg") or file.endswith(".png")]):
            record = {}
            filename = os.path.join(img_dir, filename)
            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            record["annotations"] = []
            dataset_dicts.append(record)
        return dataset_dicts
    img_dir = cfg.PUBLAYNET_DATA_DIR_TEST  # Path to your test images directory from the YAML file
    DatasetCatalog.register("doclaynet_test_no_gt", lambda: get_test_dicts(img_dir))
    # Set metadata to match the 11 classes your model was trained on
    MetadataCatalog.get("doclaynet_test_no_gt").set(thing_classes=[
        "Caption", "Footnote", "Formula", "List-item", "Page-footer",
        "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
    ])
    cfg.defrost()  # Unfreeze the configuration to allow modifications
    cfg.DATASETS.TEST = ("doclaynet_test_no_gt",)
    cfg.freeze()  # Freeze the configuration again
    model = MyTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    predictions = run_inference(cfg, model)
    print(predictions)
if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("Command Line Args:", args)
    if args.debug:
        import debugpy
        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )