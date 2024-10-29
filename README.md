<h1 align="center"> Table Detection Modules</h1> 
<p align="center">A collection of table detection modules designed to accurately detect tables across various document types, such as invoices, scientific papers, and other complex layouts.</p>


## Overview
This repository provides multiple table detection models to efficiently locate and identify tables within documents of diverse formats. We aim to continuously update and improve each module. Our goal is to establish a cohesive framework that allows all models to be seamlessly called from a unified script.


## Models Implemented and Tested
Our modules currently include the following table detection models:
1. YOLO - You Only Look Once, a real-time object detection system
2. DETR - DEtection TRansformers for high-quality table detection
3. CascadeTabNet - Cascade Mask R-CNN-based model optimized for table structure recognition

## Training Dataset
Our models are primarily trained on [Doclaynet](https://github.com/DS4SD/DocLayNet), a comprehensive human-annotated document layout segmentation dataset with 80,863 pages from a wide variety of sources. Additional testing has been conducted on complex proprietary medical documents to validate robustness and versatility across various domains.

## Acknowledgments
Special thanks to the following resources and frameworks that have significantly supported our development:
1. [YOLO Documentation](https://docs.ultralytics.com/models/yolov8/)
2. [DETR by Facebook AI Research](https://github.com/facebookresearch/detr)
3. [MMdetection Toolkit](https://mmdetection.readthedocs.io/en/latest/get_started.html)

## Contact
For questions, feedback, or collaboration opportunities, please feel free to reach out:
- [Anik De](mailto:anekde@gmail.com)
- [Ritu Singh](mailto:m23cse017@iitj.ac.in)
