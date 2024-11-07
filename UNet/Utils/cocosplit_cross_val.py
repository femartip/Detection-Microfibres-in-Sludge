import json
import os
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from pycocotools.coco import COCO

"""
def split_data(train_indices, test_indices, image_ids, image_data, data):
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    common_metadata = ['info', 'licenses', 'categories']

    for cm in common_metadata:
        train_data[cm] = data[cm]
        test_data[cm] = data[cm]

    train_image_ids = set([image_ids[i] for i in train_indices])
    test_image_ids = set([image_ids[i] for i in test_indices])

    for image in image_data:
        image_id = int(image['file_name'].split('.')[0])
        if image_id in train_image_ids:
            train_data['images'].append(image)
        else:
            test_data['images'].append(image)

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id in train_image_ids:
            train_data['annotations'].append(annotation)
        else:
            test_data['annotations'].append(annotation)

    return train_data, test_data

def print_data_info(data_dict, fold):
    images_count = len(data_dict['images'])
    annotations_count = len(data_dict['annotations'])
    print(f"Number of images: {images_count}, Number of annotations: {annotations_count}")

def k_fold_data(data_dir, NUM_FOLDS=5, seed=42):
    json_file = os.path.join(data_dir, "coco_format.json")
    images_dir = os.path.join(data_dir, "images")

    data = json.load(open(json_file, 'r'))
    image_data = data['images']
    
    annotations = data['annotations']
    image_ids = [annotations[i]['image_id'] for i in range(len(annotations))]
    category_ids = [annotations[i]['category_id'] for i in range(len(annotations))]
    
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    folds = {}

    for fold, (train_indices, test_indices) in enumerate(skf.split(image_ids, category_ids)):
        print(f"Fold {fold} has {len(train_indices)} training data and {len(test_indices)} testing data")
        train_data, test_data = split_data(train_indices, test_indices, image_ids, image_data, data)
        
        # Include annotations in the split
        train_annotations = [annotations[i] for i in train_indices]
        test_annotations = [annotations[i] for i in test_indices]

        train_data['annotations'] = train_annotations
        test_data['annotations'] = test_annotations
        
        #print(f"Data info for training data fold {fold}:")
        #print_data_info(train_data, fold)
        #print(f"Data info for testing data fold {fold}:")
        #print_data_info(test_data, fold)
        folds[fold] = {'train': train_data, 'test': test_data}
    return folds
"""
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
