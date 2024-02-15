# Deployment for Detectron2 model, load model and save it as a torchscript model

import torch
import os
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2.modeling import build_model, GeneralizedRCNN
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import TracingAdapter, dump_torchscript_IR

cfg = get_cfg()
cfg.merge_from_file("./Fine_tuned_Detectron2/models/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join("./Fine_tuned_Detectron2/models/train1238_6/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

test_image = cv2.imread("./Fine_tuned_Detectron2/data/Dataset/images/0.jpg")
test_image = torch.as_tensor(test_image.astype("float32").transpose(2, 0, 1))
print("test_image shape: ", test_image.shape)

inputs = [{"image": test_image}]

if isinstance(model, GeneralizedRCNN):
    print("is instance of GeneralizedRCNN")
    def inference(model, inputs):
        # use do_postprocess=False so it returns ROI mask
        inst = model.inference(inputs, do_postprocess=False)[0]
        return [{"instances": inst}]
else:
    inference = None

traceable_model = TracingAdapter(model, inputs, inference)
traceable_model.eval()

trace_model = torch.jit.trace(traceable_model, (test_image,))
trace_model.save("./models/model_final.pt")
print("Model saved as torchscript model")
