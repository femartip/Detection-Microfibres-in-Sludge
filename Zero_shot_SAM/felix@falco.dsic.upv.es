import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import random
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
import math
from progress.bar import Bar
from webcolors import rgb_to_name
import torchmetrics

def extract_color(img, mask):
    roi = cv2.bitwise_and(img, img, mask=mask)
    average_color = np.mean(roi, axis=(0,1))
    average_color_int = np.round(average_color).astype(int)
    try:
        color_label = rgb_to_name(average_color_int)
    except ValueError:
        return "Unknown"
    #color_ranges = {
    #        (0,0,255): "Red",
    #        (0,255,0): "Green",
    #        (255,0,0): "Blue"
    #        }
    #closest_color = min(color_ranges, key=lambda x: np.linalg.norm(average_color_int - x))
    #color_label = color_ranges[closest_color]
    return color_label

def put_text(image,text, x, y):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = max(0, min(x,image.shape[1] - text_size[0]))
    y = max(0,min(y, image.shape[0])) 
    cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
sam.to(DEVICE)

iou_metric = torchmetrics.classification.BinaryJaccardIndex(threshold=0.5)
mean_iou = []

bar = Bar('Processing', max=1660)
for file in os.listdir("./data/Dataset/images"):
    if file.endswith(".jpg"):
        #print("Processing file: {}".format(file))
        image_bgr = cv2.imread(os.path.join("./data/Dataset/images", file))
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ground_truth_mask = cv2.imread(os.path.join("./data/Dataset/ground_truth", file), cv2.IMREAD_GRAYSCALE)
        binary_ground_truth_mask = (ground_truth_mask > 175).int()
        #print("Generating mask...")
        mask_generator = SamAutomaticMaskGenerator(sam)
        mask = mask_generator.generate(img)
        #print("Mask generated")
        more_masks = False
        for n, maskn in enumerate(mask):
            #print("Processing mask " + str(n) + ":")
            bbox = maskn['bbox']  
            x, y, w, h = bbox
            w = (w*3500)/img.shape[1]
            h = (h*2625)/img.shape[0] 
            #print("x= " + str(x) + ",y= " + str(y) + ",w= " + str(w) + ",h= " + str(h))    
            if ((h/w > 3 and h > 13 and h < 5000) or (w/h>3 and w > 13 and w < 5000)) and not(w > 675 and w < 690 and h > 40 and h < 60 ):
                #print("Fibre detected")
                #print(more_masks)
                if more_masks == True:
                    prev_seg_mask_grey_text = seg_mask_grey_text
                #rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='r',fill=False)
                seg_mask = maskn['segmentation']
                seg_mask_text = seg_mask.astype(np.uint8)*255
                seg_mask_grey_text = cv2.cvtColor(seg_mask_text, cv2.COLOR_GRAY2RGB)
                #seg_mask_rgb = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
                #overlay = np.where(seg_mask_rgb != 0, seg_mask_rgb, img)
                #result = np.concatenate((seg_mask_rgb, img), axis=1)
                color = extract_color(img, seg_mask)
                w_trunc = math.trunc(w)
                h_trunc = math.trunc(h)
                text = "{}mic,{}mic, {}".format(w_trunc,h_trunc,color)
                put_text(seg_mask_grey_text, text, x, y)
                #cv2.putText(seg_mask_rgb, "{}mic,{}mic, {}".format(w_trunc,h_trunc,color), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                if more_masks == True:
                    seg_mask_grey = cv2.addWeighted(prev_seg_mask_grey_text, 0.5, seg_mask_grey_text, 0.5,0)
                    seg_mask_eval = cv2.addWeighted(seg_mask_eval, 0.5, seg_mask, 0.5,0)
                else: 
                    more_masks = True
            #print("Mask processed")

        if more_masks:
            metric = iou_metric(seg_mask_eval, binary_ground_truth_mask)
            mean_iou.append(metric.item())
            print("IOU: " + str(metric.item()))
            result = cv2.addWeighted(img, 0.7, seg_mask_grey, 0.3,0)
            cv2.imwrite(os.path.join("./fibras_detectadas/" , file), result)
    bar.next()
bar.finish()
