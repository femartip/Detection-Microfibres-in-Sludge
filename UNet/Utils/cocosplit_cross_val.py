import json
import os
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from pycocotools.coco import COCO

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

    for ann_id in coco.getAnnIds():
        annotation = coco.loadAnns(ann_id)[0]
        mask = coco.annToMask(annotation)
        if annotation['image_id'] in train_image_ids:
            train_data['annotations'].append(mask)
        elif annotation['image_id'] in test_image_ids:
            test_data['annotations'].append(mask)

    return train_data, test_data


def k_fold_data(data_dir, NUM_FOLDS=5, seed=42):
    json_file = os.path.join(data_dir, "coco_format.json")
    images_dir = os.path.join(data_dir, "images")
    
    coco = COCO(json_file)
    data = coco.dataset
    annotations = data['annotations']
    
    # Group by image and category
    image_ids_by_category = []
    for ann in annotations:
        image_ids_by_category.append((ann['image_id'], ann['category_id']))

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    folds = {}

    for fold, (train_indices, test_indices) in enumerate(skf.split(image_ids_by_category, [cat for _, cat in image_ids_by_category])):
        print(f"Fold {fold} has {len(train_indices)} training data and {len(test_indices)} testing data")

        # Collect unique image_ids for train and test sets based on indices
        train_image_ids = [image_ids_by_category[i][0] for i in train_indices]
        test_image_ids = [image_ids_by_category[i][0] for i in test_indices]

        # Split data by category and prepare training/testing data
        train_data, test_data = split_data_by_category(train_image_ids, test_image_ids, coco)

        folds[fold] = {'train': train_data, 'test': test_data}

    return folds
