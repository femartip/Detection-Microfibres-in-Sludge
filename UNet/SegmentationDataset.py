import os
import json
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, images_dir, coco):
        self.images = images
        self.masks = masks
        self.images_dir = images_dir
        self.coco = coco

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_shape = (image_info['height'], image_info['width'])
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_info['id'])):
            mask += self.coco.annToMask(ann)
        
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask

def split_data_by_category(train_indices, test_indices, coco):
    train_data = defaultdict(list)
    test_data = defaultdict(list)

    train_image_ids = set(train_indices)
    test_image_ids = set(test_indices)

    for img_id in coco.getImgIds():
        image = coco.loadImgs(img_id)[0]
        if img_id in train_image_ids:
            train_data['images'].append(image)
        elif img_id in test_image_ids:
            test_data['images'].append(image)

    return train_data, test_data

def k_fold_data(data_dir, NUM_FOLDS=5, seed=42):
    json_file = os.path.join(data_dir, "coco_format.json")
    images_dir = os.path.join(data_dir, "images")
    
    coco = COCO(json_file)
    data = coco.dataset
    annotations = data['annotations']
    
    image_ids_by_category = []
    for ann in annotations:
        image_ids_by_category.append((ann['image_id'], ann['category_id']))

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    folds = []

    for fold, (train_indices, test_indices) in enumerate(skf.split(image_ids_by_category, [cat for _, cat in image_ids_by_category])):
        print(f"Fold {fold} has {len(train_indices)} training data and {len(test_indices)} testing data")

        # Collect unique image_ids for train and test sets based on indices
        train_image_ids = [image_ids_by_category[i][0] for i in train_indices]
        test_image_ids = [image_ids_by_category[i][0] for i in test_indices]

        # Split data by category and prepare training/testing data
        train_data, test_data = split_data_by_category(train_image_ids, test_image_ids, coco)

        #folds[fold] = {'train': train_data, 'test': test_data}
        train_dataset = SegmentationDataset(train_data['images'], train_data['annotations'], images_dir, coco)
        test_dataset = SegmentationDataset(test_data['images'], test_data['annotations'], images_dir, coco)

        folds.append((train_dataset, test_dataset))

    return folds

def get_k_fold_datasets(data_dir, num_folds=5, seed=42):
    return k_fold_data(data_dir, NUM_FOLDS=num_folds, seed=seed)
"""
    datasets = {}
    
    for fold, fold_data in folds.items():
        train_images = fold_data['train']['images']
        train_masks = fold_data['train']['annotations']
        test_images = fold_data['test']['images']
        test_masks = fold_data['test']['annotations']
        
        # Create datasets
        train_dataset = KFoldSegmentationDataset(train_images, train_masks, os.path.join(data_dir, 'images'))
        test_dataset = KFoldSegmentationDataset(test_images, test_masks, os.path.join(data_dir, 'images'))
        
        datasets[fold] = {'train': train_dataset, 'test': test_dataset}
    
    return datasets
"""

