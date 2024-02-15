import torch
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import random
import matplotlib.pyplot as plt
import numpy as np
import math
#from progress.bar import Bar
from webcolors import rgb_to_name
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

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
    

#DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

cfg_inst = get_cfg()
cfg_inst.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_inst.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
# Find a model from detectron2's model zoo.  https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
cfg_inst.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg_inst)

#bar = Bar('Processing', max=1660)
for file in os.listdir("./fibras_low_res_all"):
    if file.endswith(".jpg"):
        print("Processing file: {}".format(file))
        image_bgr = cv2.imread(os.path.join("./fibras_low_res_all/", file))
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print("Generating mask...")
        outputs = predictor(img)
        N,_,_ = outputs['instances'].pred_masks.shape
        print("Mask generated")
        more_masks = False
        for n in range(N):
            print("Processing mask " + str(n) + ":")
            maskn = outputs['instances'].pred_masks[n]
            x1, y1, x2, y2 = outputs['instances'].pred_boxes[n].tensor.cpu().tolist()[0] 
            w = x2 - x1
            h = y2 - y1
            w = (w*3500)/img.shape[1]
            h = (h*2625)/img.shape[0] 
            print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1,y1,x2,y2))   
            if h/w > 3 and h > 13 and h < 5000:
                print("Fibre detected")
                #print(more_masks)
                if more_masks == True:
                    prev_seg_mask_rgb = seg_mask_rgb
                #rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='r',fill=False)
                seg_mask = maskn.cpu().numpy()
                seg_mask_rgb = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
                #overlay = np.where(seg_mask_rgb != 0, seg_mask_rgb, img)
                #result = np.concatenate((seg_mask_rgb, img), axis=1)
                color = extract_color(img, seg_mask)
                w_trunc = math.trunc(w)
                h_trunc = math.trunc(h)
                text = "{}mic,{}mic, {}".format(w_trunc,h_trunc,color)
                put_text(seg_mask_rgb, text, x, y)
                #cv2.putText(seg_mask_rgb, "{}mic,{}mic, {}".format(w_trunc,h_trunc,color), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                if more_masks == True:
                    seg_mask_rgb = cv2.addWeighted(prev_seg_mask_rgb, 0.5, seg_mask_rgb, 0.5,0)
                else: 
                    more_masks = True
            print("Mask processed")

        if more_masks:
            result = cv2.addWeighted(img, 0.7, seg_mask_rgb, 0.3,0)
            cv2.imwrite(os.path.join("./fibras_detectadas/" , file), result)
    #bar.next()
#bar.finish()
