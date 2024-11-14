import os
import json
import argparse
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

NUM_FOLDS = 5
SEED = 42

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
        elif image_id in test_image_ids:
            test_data['images'].append(image)

    train_data['annotations'] = data['annotations']
    test_data['annotations'] = data['annotations']
    """
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id in train_image_ids:
            train_data['annotations'].append(annotation)
        elif image_id in test_image_ids:
            test_data['annotations'].append(annotation)
    """
    return train_data, test_data

def print_data_info(data_dict, fold):
    images_count = len(data_dict['images'])
    annotations_count = len(data_dict['annotations'])
    print(f"Number of images: {images_count}, Number of annotations: {annotations_count}")

def k_fold_data(image_ids, category_ids, image_data, data, dir_path):
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    pairs = []
    for fold, (train_indices, test_indices) in enumerate(skf.split(image_ids, category_ids)):
        print(f"Fold {fold} has {len(train_indices)} training data and {len(test_indices)} testing data")
        train_data, test_data = split_data(train_indices, test_indices, image_ids, image_data, data)
        train_file = os.path.join(dir_path, f"train_coco_{fold}_fold.json")
        test_file = os.path.join(dir_path, f"test_coco_{fold}_fold.json")
        with open(train_file, 'w') as train_file:
            json.dump(train_data, train_file)
        with open(test_file, 'w') as test_file:
            json.dump(test_data, test_file)
        print(f"Data info for training data fold {fold}:")
        print_data_info(train_data, fold)
        print(f"Data info for testing data fold {fold}:")
        print_data_info(test_data, fold)
        pairs.append([train_file, test_file])
    
    return pairs 


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="./Fine_tuned_Detectron2/data/Dataset/Dataset_vidrio")
    args = args.parse_args()
    dir_path = args.data_dir

    data = json.load(open(os.path.join(dir_path, "coco_format.json"))) #Load data
   
    image_data = data['images'] #Get image data
    annotations = data['annotations'] #Get annotations
    
    # Get unique image IDs and create a mapping of image ID to file name
    image_ids = [annotations[i]['image_id'] for i in range(len(annotations))]
    # Get category ID for each image
    category_ids = [annotations[i]['category_id'] for i in range(len(annotations))]

    pairs = k_fold_data(image_ids, category_ids, image_data, data, dir_path) #Split data into k folds
        

if __name__ == "__main__":
    main()