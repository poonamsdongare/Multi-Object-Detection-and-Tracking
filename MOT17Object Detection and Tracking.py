# -*- coding: utf-8 -*-
"""
"""
#%% Import of Libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import torch
import numpy as np
from sort import Sort  # SORT Tracker


#%%
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#%%
# # Convert gt.txt to YOLO format
# def convert_gt_to_yolo(base_dir):
#     for root, _, files in os.walk(base_dir):
#         if 'gt' in root:
#             gt_file = os.path.join(root, 'gt.txt')
#             img_folder = root.replace('gt', 'img1')

#             # Create labels folder (YOLO expects labels next to images)
#             labels_folder = os.path.join(os.path.dirname(root), 'labels')
#             os.makedirs(labels_folder, exist_ok=True)

#             with open(gt_file, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     frame, id, x, y, w, h, _, _, _ = map(float, line.strip().split(','))

#                     # YOLO format: class_id center_x center_y width height (normalized)
#                     img_path = os.path.join(img_folder, f'{int(frame):06}.jpg')
#                     img = cv2.imread(img_path)
#                     img_h, img_w = img.shape[:2]

#                     center_x = (x + w / 2) / img_w
#                     center_y = (y + h / 2) / img_h
#                     width = w / img_w
#                     height = h / img_h

#                     label_file = os.path.join(labels_folder, f'{int(frame):06}.txt')
#                     with open(label_file, 'a') as lf:
#                         lf.write(f"0 {center_x} {center_y} {width} {height}\n")

#             print(f"Converted ground truth in {root} to YOLO format.")

# # Call this for both train and test sets
# convert_gt_to_yolo(train_dir)
# convert_gt_to_yolo(test_dir)

#%%
# Finetune the YOLOv5 model 
# def train_yolo(data_yaml, epochs=50):
#     model.train(data=data_yaml, epochs=epochs)
#     model.save('trained_yolo.pt')
    
#%% 
# Initialize SORT tracker
sort_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

#%%
# Read images from dataset
def read_images(base_dir):
    image_data = {}
    for root, _, files in os.walk(base_dir):
        if 'img1' in root:
            folder_name = os.path.basename(os.path.dirname(root))
            images = sorted([os.path.join(root, f) for f in files if f.endswith('.jpg')])
            image_data[folder_name] = images
    return image_data
#%% 
# Load ground truth data
def load_ground_truth(base_dir):
    gt_data = {}
    for root, _, files in os.walk(base_dir):
        if 'gt' in root:
            for file in files:
                if file == 'gt.txt':
                    folder_name = os.path.basename(os.path.dirname(root))
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f.readlines():
                            values = line.strip().split(',')
                            obj_id, x, y = values[1], float(values[2]), float(values[3])
                            gt_data[f"{folder_name}_{obj_id}"] = (x, y)
    return gt_data

#%%
# Object tracking with SORT
def track_objects(images):
    predictions = {}  # Store predictions for MOTA evaluation

    for img_path in images:
        img = cv2.imread(img_path)
        results = model(img)
        detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls

        # Format detections for SORT (x1, y1, x2, y2, conf)
        sort_detections = np.array([[x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf, _ in detections])

        # Update SORT tracker
        tracked_objects = sort_tracker.update(sort_detections)

        # Store predictions with stable IDs
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            predictions[track_id] = (center_x, center_y)

            # Draw bounding boxes and track IDs
            cv2.circle(img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f'ID: {int(track_id)}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show tracking result
        cv2.imshow('Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return predictions

#%%
# MOTA Calculation
def calculate_mota(ground_truth, predictions):
    fn = len(set(ground_truth.keys()) - set(predictions.keys()))
    fp = len(set(predictions.keys()) - set(ground_truth.keys()))
    id_switches = sum(1 for k in ground_truth if k in predictions and ground_truth[k] != predictions[k])
    total_gt = len(ground_truth)
    mota = 1 - (fn + fp + id_switches) / total_gt if total_gt > 0 else 0
    return mota

#%%

# Paths to dataset
train_dir = r'C:\Users\Poonam\Desktop\DL Project\MOT17\train'
test_dir = r'C:\Users\Poonam\Desktop\DL Project\MOT17\test'

# Load images and ground truth
ground_truth = load_ground_truth(train_dir)
train_images = read_images(train_dir)
test_images = read_images(test_dir)

# Process train images
for folder, images in train_images.items():
    print(f"Processing {folder} with {len(images)} images...")
    predictions = track_objects(images)
    mota = calculate_mota(ground_truth, predictions)
    print(f"MOTA score for {folder}: {mota}")

# Process test images
for folder, images in test_images.items():
    print(f"Processing {folder} with {len(images)} images...")
    track_objects(images)
