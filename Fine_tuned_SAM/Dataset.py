import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import json
from PIL import Image
from patchify import patchify
from datasets import Dataset

def json_to_mask(json_file, image_shape=(750, 1000,3)):
    mask = np.zeros(image_shape, dtype=np.uint8)
    #print(mask.shape)
    #print(json_file[0]['content'])
    pts = []
    if len(json_file[0]['content']) == 0:
        return mask
    for point in json_file[0]['content']:
        if json_file[0]['contentType'] == 'polygon':
            #print(point['x'], point['y'])
            x,y = int(point['x']), int(point['y'])
            #print(mask[y,x])
            pts.append([x,y])     
    #print(mask.shape)
    pts = np.array(pts, np.int32)
    mask = cv2.fillPoly(mask, pts=[pts], color=(255,255,255))    
    return mask

def patchify_dataset(images, masks, patch_size=(256,256, 3), step=256):
    #Return patchified images and masks as numpy arrays
    patch_images = []
    patch_masks = []
    for image, mask in zip(images, masks):
        #print(image.shape)
        #print(mask.shape)
        patch_images.append(patchify(image, patch_size, step))
        patch_masks.append(patchify(mask, patch_size, step))
    patch_images = np.array(patch_images)
    patch_masks = np.array(patch_masks)
    patch_images = np.reshape(patch_images, (-1, patch_size[0], patch_size[1], 3))
    patch_masks = np.reshape(patch_masks, (-1, patch_size[0], patch_size[1], 3))
    return patch_images, patch_masks  

def main():
    images = [cv2.imread(file) for file in glob.glob("./data/Dataset/images/*.jpg")]
    #images = [Image.fromarray(cv2.imread(file)) for file in glob.glob("./Dataset/data/*.jpg")]
    masks = [json_to_mask(json.load(open(file))) for file in glob.glob("./data/Dataset/masks/*.json")]

    images = np.array(images)
    masks = np.array(masks)

    #print(images.shape)
    #print(masks.shape)
    #print(images[0].dtype)
    #print(masks[0].dtype)

    patch_images, patch_masks = patchify_dataset(images, masks)
    #print(patch_images.shape)
    #print(patch_masks.shape)
    """
    valid_indices = [i for i, patch_mask in enumerate(patch_masks) if patch_masks.max() != 0]
    filtered_images = patch_images[valid_indices]
    filtered_masks = patch_masks[valid_indices]
    print("Image shape:", filtered_images.shape)
    print("Mask shape:", filtered_masks.shape)
    """
    filtered_images = patch_images
    filtered_masks = patch_masks

    dataset_dict = {
        "image": [Image.fromarray(image) for image in filtered_images],
        "mask": [Image.fromarray(mask) for mask in filtered_masks]
    }

    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk("train_929img.hf")


if __name__ == "__main__":
    main()