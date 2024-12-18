import os
import json
from collections import defaultdict
import logging
from sklearn.model_selection import StratifiedKFold
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import datasets
from PIL import Image, ImageDraw

class CocoMaskDataset(datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CocoMaskDataset, self).__init__(root, annFile, transform)
        self.coco = COCO(annFile)  # Load COCO API
        self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

    def __getitem__(self, index):
        # Load image and target using parent class
        img, targets = super(CocoMaskDataset, self).__getitem__(index)

        # Convert image to tensor using transform
        if self.transform:
            img = self.transform(img)
        if type(img) == Image.Image:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

        # Generate binary masks and bounding boxes for the annotations
        bin_mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
        bboxes = []
        for target in targets:
            segmentation = target['segmentation']
            bbox = target['bbox']

            height, width = img.shape[1:]
            
            if isinstance(segmentation, list):  # Polygon format
                rle = coco_mask.frPyObjects(segmentation, height, width)  # Convert to RLE
                binary_mask = coco_mask.decode(rle)  # Decode RLE to binary mask

                # Handle multi-part objects (sum along axis 2 if necessary)
                if len(binary_mask.shape) == 3:
                    binary_mask = np.any(binary_mask, axis=2).astype(np.uint8)
            else:  # If segmentation is already in RLE format
                binary_mask = coco_mask.decode(segmentation)

            bin_mask += binary_mask
            bboxes.append(bbox)

        bin_mask = torch.from_numpy(bin_mask).float()
        bin_mask /= bin_mask.max()
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        
        return img, bin_mask

    def _polygon_to_mask(self, polygons, image_size):
        """Convert polygon segmentation to a binary mask."""
        mask = Image.new('L', image_size, 0)
        for polygon in polygons:
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        return np.array(mask, dtype=np.uint8)


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


