from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import cv2

model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
mp_sam_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
mp_sam_model.load_state_dict(torch.load("./Models/sam_929img_10ep_checkpoint.pth"))

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
mp_sam_model.to(device)

for j in range(930,1660):
    print(j)
    image = cv2.imread("./data/Dataset/images/{}.jpg".format(j))
    image = np.array(image)
    #print(image.shape)
    patches = patchify(image, (256, 256, 3), step=256)
    patches = np.reshape(patches, (patches.shape[0]*patches.shape[1], 256, 256, 3))
    #print(patches.shape)

    array_size = 256
    grid_size = 30
    x = np.linspace(0, array_size-1, grid_size)
    y = np.linspace(0, array_size-1, grid_size)
    xv, yv = np.meshgrid(x, y)
    xv_list = xv.tolist()
    yv_list = yv.tolist()
    input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
    input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

    pred = []
    for i in range(patches.shape[0]):
        patch = patches[i]
        #print(patch.shape)
        single_patch = Image.fromarray(patch)

        inputs = processor(single_patch, input_points=input_points, return_tensors="pt")
        #print(inputs.keys())

        inputs = {k: v.to(device) for k, v in inputs.items()}
        mp_sam_model.eval()

        with torch.no_grad():
            outputs = mp_sam_model(**inputs, multimask_output=False)
        #print(outputs.keys())

        single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

        single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
        single_patch_prediction = (single_patch_prob > 0.5).astype(np.uint8)
        pred.append(single_patch_prediction)

    #for i in range(len(pred)):
    #    cv2.imwrite("./1038_{}.jpg".format(i), pred[i])
    pred = np.array(pred)
    #print(pred.shape)
    joined_pred_1 = cv2.hconcat([pred[0], pred[1], pred[2]])
    joined_pred_2 = cv2.hconcat([pred[3], pred[4], pred[5]])
    joined_pred = cv2.vconcat([joined_pred_1, joined_pred_2])
    #print(joined_pred.shape)
    cv2.imwrite("./data/predictions/{}.jpg".format(j), joined_pred)