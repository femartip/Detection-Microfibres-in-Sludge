import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import torch
import glob
import json
from PIL import Image
from patchify import patchify
from PIL import Image
from datasets import Dataset
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import torchmetrics

def json_to_mask(json_file, image_shape=(750, 1000)):
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
    mask = cv2.fillPoly(mask, pts=[pts], color=255)
    return mask

def patchify_dataset(images, patch_size=(256,256, 3), step=256):
    #Return patchified images and masks as numpy arrays
    patch_images = []
    for i in range(images.shape[0]):
        #print(images[i].shape)
        #print(mask.shape)
        patch_images.append(patchify(images[i], patch_size, step))
        #patch_masks.append(patchify(mask, patch_size, step))
    patch_images = np.array(patch_images)
    #patch_masks = np.array(patch_masks)
    if len(patch_size) == 3:
      patch_images = np.reshape(patch_images, (-1, patch_size[0], patch_size[1], patch_size[2]))
    elif len(patch_size) == 2:
      patch_images = np.reshape(patch_images, (-1, patch_size[0], patch_size[1]))
    return patch_images

def get_bounding_box(ground_truth_map):
    #print(ground_truth_map)
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):

    item = self.dataset[idx]
    #print(item)
    image = item["image"]

    if type(item["mask"]) == list:
      ground_truth_mask = [np.array(i) for i in item["mask"]]
      prompt = [get_bounding_box(mask) for mask in ground_truth_mask]
    else:
      ground_truth_mask = np.array(item["mask"], np.float32)
      prompt = get_bounding_box(ground_truth_mask)



    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
    #inputs = self.processor(image, input_boxes=[[]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

def dataset():
  image_files = sorted(glob.glob("./drive/MyDrive/Dataset/images/*.jpg"))
  mask_files = sorted(glob.glob("./drive/MyDrive/Dataset/masks/*.json"))

  images = [cv2.imread(file) for file in image_files]
  masks = [json_to_mask(json.load(open(file))) for file in mask_files]

  images = np.array(images)
  masks = np.array(masks)

  #print(images.shape)
  #print(masks.shape)
  #print(images[0].dtype)
  #print(masks[0].dtype)

  #plt.imshow(images[0])
  plt.imshow(masks[0])

  #patch_images, patch_masks = patchify_dataset(images, masks)
  patch_images = patchify_dataset(images)
  patch_masks = patchify_dataset(masks, patch_size=(256,256))
  #print(patch_images.shape)
  #print(patch_masks.shape)

  valid_indices = [i for i, patch_mask in enumerate(patch_masks) if patch_mask.max() != 0]
  filtered_images = patch_images[valid_indices]
  filtered_masks = patch_masks[valid_indices]
  #print("Image shape:", filtered_images.shape)
  #print("Mask shape:", filtered_masks.shape)

  #filtered_images = patch_images
  #filtered_masks = patch_masks

  dataset_dict = {
      "image": [Image.fromarray(image) for image in filtered_images],
      "mask": [Image.fromarray(mask, mode='L') for mask in filtered_masks]
  }

  dataset = Dataset.from_dict(dataset_dict)
  #dataset.save_to_disk("train_929img_new")
  return dataset





def train(dataset):
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  model = SamModel.from_pretrained("facebook/sam-vit-base")

  for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
      param.requires_grad_(False)

  train_dataset = SAMDataset(dataset=dataset, processor=processor)
  #example = train_dataset[0]
  #for k,v in example.items():
    #print("key:", k, "shape:", v.shape)

  train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
  #batch = next(iter(train_dataloader))
  #for k,v in batch.items():
    #print("key:", k, "shape:", v.shape)

  optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
  seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') #Try DiceFocalLoss, FocalLoss, DiceCELoss

  num_epochs = 5
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  mean_losses = []
  iou_metric = torchmetrics.classification.BinaryJaccardIndex(threshold=0.5).to(device)
  iou_scores = []

  model.train()
  for epoch in range(num_epochs):
      epoch_losses = []
      epoch_iou_scores = []
      for batch in tqdm(train_dataloader):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].unsqueeze(1).to(device),
                        multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        binary_ground_truth_masks = (ground_truth_masks > 127.5).int()
        metric = iou_metric(predicted_masks, binary_ground_truth_masks.unsqueeze(1))
        epoch_iou_scores.append(metric.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        epoch_losses.append(loss.item())

      iou_scores.append(mean(epoch_iou_scores))
      mean_losses.append(mean(epoch_losses))
      print(f'EPOCH: {epoch}')
      print(f'Mean loss: {mean(epoch_losses)}')
      print(f'Mean IoU score: {mean(epoch_iou_scores)}')

  plt.plot(mean_losses)
  plt.title("Mean losses")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.savefig("./sam_929img_{}_losses.png".format(num_epochs))
  plt.clf()

  plt.plot(iou_scores)
  plt.title("Mean IoU scores")
  plt.xlabel("Epoch")
  plt.ylabel("IoU score")
  plt.savefig("./sam_929img_{}_iou.png".format(num_epochs))
  plt.clf()

  torch.save(model.state_dict(), "./sam_929img_{}_checkpoint.pth".format(num_epochs))

if __name__ == "__main__":
  dataset = dataset()
  train(dataset)