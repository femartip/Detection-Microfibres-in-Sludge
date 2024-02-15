import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from matplotlib import pyplot as plt
import os
import numpy as np
import torch

cfg = get_cfg()
cfg.merge_from_file("./Fine_tuned_Detectron2/models/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join("./Fine_tuned_Detectron2/models/model_final.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(cfg)
model.to(DEVICE)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

#plt.ion()

for j in range(930,1660):
    model.eval()
    img = cv2.imread("./data/Dataset/images/{}.jpg".format(j))  #BGR image

    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        outputs = model([inputs])
    #print(outputs)

    pred_masks = outputs[0]["instances"].pred_masks.cpu().numpy()
    #print(pred_masks)

    if pred_masks.shape[0] == 0:
        continue
    pred_boxes = outputs[0]["instances"].pred_boxes.tensor.cpu().numpy()
    x1,y1,x2,y2 = pred_boxes[0]
    x1,y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    pred_masks = pred_masks.squeeze(1)
    #print(pred_masks.shape)

    #print(image)
    #Show the original image to the left and the heatmap to the right
    plt.subplot(1,pred_masks.shape[0]+1,1)
    plt.imshow(img)
    plt.title("Original image {}".format(j))
    for i in range(pred_masks.shape[0]):
        plt.subplot(1,pred_masks.shape[0]+1,i+2)
        plt.imshow(pred_masks[i] , cmap='hot', interpolation='nearest')

    plt.colorbar()
    plt.title("Heatmap of image {}".format(j))
    #plt.savefig("./Fine_tuned_Detectron2/data/heatmap_mask/{}_{}.png".format(j,i))
    
    plt.show()  
    #plt.pause(2)
    #plt.close()
    