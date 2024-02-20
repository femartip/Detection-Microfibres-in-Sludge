import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model, RPN_HEAD_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead

import os
import datetime
import numpy as np
from scipy.spatial import distance
from typing import List
import torch
from matplotlib import pyplot as plt

"""
Apply trained model on inference images to get objectness of rpn detectrions
For this, need to change file ./detectron2/modeling/proposal_generator/rpn.py
This saves the probability map of object existance for each level of the FPN
Got the ide form https://medium.com/@hirotoschwert/digging-into-detectron-2-part-4-3d1436f91266
"""
@RPN_HEAD_REGISTRY.register()
class MyRPNHead(StandardRPNHead):
    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        i=0
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
            ################################################################################3
            o = pred_objectness_logits[-1].sigmoid() * 255
            o = o.cpu().detach().numpy()[0, 1]
            o = cv2.resize(o, (320, 184))
            now = datetime.datetime.now()
            cv2.imwrite('./Fine_tuned_Detectron2/data/objectness/' + str(i)+'.png', np.asarray(o, dtype=np.uint8))
            i+=1
            #############################################################################
        return pred_objectness_logits, pred_anchor_deltas
    


model_path = "49_final"

cfg = get_cfg()
cfg.merge_from_file("./Fine_tuned_Detectron2/models/{}/config.yaml".format(model_path))
cfg.MODEL.WEIGHTS = os.path.join("./Fine_tuned_Detectron2/models/{}/model_final.pth".format(model_path))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.RPN.HEAD_NAME = "MyRPNHead"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(cfg)
model.to(DEVICE)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

files = os.listdir('./Fine_tuned_Detectron2/data/inference')
files = [file.split('.')[0] for file in files]

for j,file in enumerate(files):
    
    model.eval()
    imgs = []
    img = cv2.imread("./Fine_tuned_Detectron2/data/inference/{}.jpg".format(file))  #BGR image
    img = cv2.resize(img, (1000, 750))
    imgs.append(img)
    
    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        outputs = model([inputs])

    for i in range(5):  # five levels
        objectness = cv2.imread('./Fine_tuned_Detectron2/data/objectness/' + str(i) + '.png')
        os.rename('./Fine_tuned_Detectron2/data/objectness/' + str(i) + '.png', './Fine_tuned_Detectron2/data/objectness/'+ str(j) + "_" + str(i) + '.png')
        heatmap = cv2.applyColorMap(objectness, cv2.COLORMAP_JET)
        imgs.append(cv2.resize(imgs[0], (320, 184)) // 2 + heatmap // 2)  # blending
    fig = plt.figure(figsize=(16, 7))
    for i, img in enumerate(imgs):
        fig.add_subplot(2, 3, i + 1)
        if i > 0:
            plt.imshow(img[0:-1, :, ::-1])  # ::-1 removes the edge
            plt.title("objectness on P" + str(i + 1))
        else:
            plt.imshow(img[:, :, ::-1])
            plt.title("input image")
    plt.savefig("./Fine_tuned_Detectron2/data/heatmap_ROI/{}.png".format(j))
    plt.close()
    

print("Done")