import os
import json
from collections import defaultdict
import logging
from sklearn.model_selection import StratifiedKFold
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
class SegmentationDataset(Dataset):
    def __init__(self, coco, images_dir, image_ids):
        self.coco = coco
        self.images_dir = images_dir
        self.image_ids = image_ids

        logging.info(f"Creating dataset with {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_shape = (image_info['height'], image_info['width'])
        mask = np.zeros(mask_shape, dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()
        
        return image, mask


def get_k_fold_dataset(data_dir, NUM_FOLDS=5, seed=42):
    logging.basicConfig(level=logging.INFO)
    json_file = os.path.join(data_dir, "coco_format.json")
    coco = COCO(json_file)
    images_dir = os.path.join(data_dir, "images")
    data = coco.dataset
    annotations = data['annotations']
    
    image_ids_by_category = [(ann['image_id'], ann['category_id']) for ann in annotations]

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    folds = [([image_ids_by_category[i][0] for i in train_indices],[image_ids_by_category[i][0] for i in test_indices])for train_indices, test_indices in skf.split(image_ids_by_category, [cat for _, cat in image_ids_by_category])]

    return folds


