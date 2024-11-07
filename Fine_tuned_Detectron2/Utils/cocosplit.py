import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def split_dataset(coco, args, save: bool = False):
    info = coco['info']
    licenses = coco['licenses']
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']
    number_of_images = len(images)
    print(number_of_images)

    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
    if args.having_annotations:
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

    if args.multi_class:
        annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)
        annotation_categories =  funcy.lremove(lambda i: annotation_categories.count(i) <=1  , annotation_categories)

        annotations =  funcy.lremove(lambda i: i['category_id'] not in annotation_categories  , annotations)

        if args.split < 1:
            X_train, y_train, X_test, y_test = iterative_train_test_split(np.array([annotations]).T,np.array([ annotation_categories]).T, test_size = 1-args.split)
            print("Saved {} entries in {} and {} in {}".format(len(X_train), args.train, len(X_test), args.test))
            if save:
                save_coco(args.train, info, licenses, filter_images(images, X_train), X_train, categories)
                save_coco(args.test, info, licenses, filter_images(images, X_test), X_test, categories)
            return X_train, y_train, X_test, y_test
        elif args.split == 1:
            print("Saved {} entries in {}".format(len(annotations), args.train))
            if save:
                save_coco(args.train, info, licenses, images, annotations, categories)
            return images, annotations, None, None        
        
    elif args.split < 1:
        X_train, X_test = train_test_split(images, train_size=args.split)
        anns_train = filter_annotations(annotations, X_train)
        anns_test = filter_annotations(annotations, X_test)
        print("Saved {} entries in {} and {} in {}".format(len(anns_train), args.train, len(anns_test), args.test))
        if save:
            save_coco(args.train, info, licenses, X_train, anns_train, categories)
            save_coco(args.test, info, licenses, X_test, anns_test, categories)
        return X_train, anns_train, X_test, anns_test
    
    elif args.split == 1:
        X_train = images
        anns_train = filter_annotations(annotations, X_train)
        print("Saved {} entries in {}".format(len(anns_train), args.train))
        if save:
            save_coco(args.train, info, licenses, X_train, anns_train, categories)
        return X_train, anns_train, None, None
    else:
        print("No data saved. You should specify either a train or a test set or set split to 1")

def main():
    parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
    parser.add_argument('annotations', metavar='coco_annotations', type=str,
                        help='Path to COCO annotations file.')
    parser.add_argument('train', type=str, help='Where to store COCO training annotations')
    parser.add_argument('-test', type=str, help='Where to store COCO test annotations')
    parser.add_argument('-s', dest='split', type=float, required=True,
                        help="A percentage of a split; a number in (0, 1)")
    parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                        help='Ignore all images without annotations. Keep only these with at least one annotation')

    parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                        help='Split a multi-class dataset while preserving class distributions in train and test sets')

    args = parser.parse_args()
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        X_train, y_train, X_test, y_test = split_dataset(coco, args, save=True)

if __name__ == "__main__":
    main()