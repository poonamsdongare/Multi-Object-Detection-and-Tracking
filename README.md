# Multi-Object-Detection-and-Tracking

## **Introduction**  
This project focuses on **multi-object detection and tracking** using the **MOT17 dataset**, a benchmark dataset for pedestrian tracking in real-world urban environments. 
The goal is to train an object detection model based on **YOLO (You Only Look Once)** to detect and track pedestrians across multiple video sequences.  

The MOT17 dataset presents various challenges, including **crowded scenes, occlusions, motion blur, and different lighting conditions**. 
By leveraging state-of-the-art deep learning techniques, we aim to improve object detection accuracy and tracking performance using **YOLO and MOTA (Multiple Object Tracking Accuracy) metrics**.  

### **Project Objectives**  
✔ Train an object detection model on MOT17 dataset annotations.  
✔ Convert the dataset into a YOLO-friendly format for training.  
✔ Evaluate tracking performance using MOTA and related metrics.  
---

## **Dataset: MOT17 Overview**  
The **MOT17 dataset** is a benchmark dataset used for **multi-object tracking (MOT)** and **object detection** in real-world surveillance environments. 
It includes **21 sequences in the training set and 21 sequences in the test set**, each containing labeled pedestrian annotations for evaluating tracking algorithms.  

### **Dataset Structure**  
The dataset is structured as follows:
```
/MOT17
├── train/
│ ├── MOT17-01-DPM/
│ ├── MOT17-02-DPM/
│ ├── MOT17-03-FRCNN/
│ ├── MOT17-04-SDP/
│ ├── ... (21 sequences)
├── test/
│ ├── MOT17-01-DPM/
│ ├── MOT17-02-DPM/
│ ├── MOT17-03-FRCNN/
│ ├── MOT17-04-SDP/
│ ├── ... (21 sequences)

Each sequence contains:
```
/MOT17-XX-[Detector]/
├── img1/ # Image frames
├── gt/ # Ground truth annotations
│ ├── gt.txt # Bounding box labels
├── det/ # Precomputed detections
├── seqinfo.ini # Metadata (FPS, frame count, etc.)


