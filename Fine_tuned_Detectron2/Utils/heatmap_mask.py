import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances, ROIMasks
from matplotlib import pyplot as plt
import os
import numpy as np
import torch

model_path = "49_final"

cfg = get_cfg()
cfg.merge_from_file("./Fine_tuned_Detectron2/models/{}/config.yaml".format(model_path))
cfg.MODEL.WEIGHTS = os.path.join("./Fine_tuned_Detectron2/models/{}/model_final.pth".format(model_path))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(cfg)
model.to(DEVICE)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)


files = os.listdir('./Fine_tuned_Detectron2/data/inference')
files = [file.split('.')[0] for file in files]

for j in files:
    model.eval()
    img = cv2.imread("./Fine_tuned_Detectron2/data/inference/{}.jpg".format(j))  #BGR image
    img = cv2.resize(img, (1000, 750))

    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        #outputs = model([inputs])
        outputs = model.inference([inputs],do_postprocess=False)
    #print(outputs)
    
    pred_masks = outputs[0].get_fields()['pred_masks'].cpu().numpy()
    #print(pred_masks)

    if pred_masks.shape[0] == 0:
        continue
    pred_boxes = outputs[0].get_fields()['pred_boxes'].tensor.cpu().numpy()
    
    pred_masks = pred_masks.squeeze(1)
    
    #Show the original image to the left and the heatmap to the right
    for i in range(pred_masks.shape[0]):
        x1,y1,x2,y2 = pred_boxes[i]
        x1,y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        copy_img = img.copy()
        cv2.rectangle(copy_img, (x1,y1), (x2,y2), (0,255,0), 2)
        fig = plt.figure(figsize=(16, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(copy_img)
        fig.add_subplot(1, 2, 2)
        plt.imshow(pred_masks[i] , cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Heatmap of image {}".format(j))
        plt.savefig("./Fine_tuned_Detectron2/data/heatmap_mask/{}_{}.png".format(j,i))
        plt.close()

print("Done")