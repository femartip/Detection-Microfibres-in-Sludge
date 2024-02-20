import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import os
from webcolors import rgb_to_name
import numpy as np
import math
from scipy.spatial import distance
from skimage.morphology import skeletonize
import torch

def extract_color(img, mask):
    roi = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imwrite("./Fine_tuned_Detectron2/data/{}.jpg".format(j), roi)
    roi = np.where(roi != 0)
    #print(roi)
    average_color = (np.mean(roi[0]), np.mean(roi[1]), np.mean(roi[2]))
    #print(average_color)
    average_color_int = np.round(average_color).astype(int)
    try:
        color_label = rgb_to_name(average_color_int)
        #print(color_label)
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

def merge_boxes_and_masks(pred_boxes, pred_masks):
    sorted_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][0])
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_masks = [pred_masks[i] for i in sorted_indices]

    # Initialize the list of merged boxes and masks
    merged_boxes = [pred_boxes[0]]
    merged_masks = [pred_masks[0]]

    for current_box, current_mask in zip(pred_boxes[1:], pred_masks[1:]):
        # Get the last box and mask in the merged_boxes and merged_masks lists
        last_box = merged_boxes[-1]
        last_mask = merged_masks[-1]

        # If the current box intersects with the last box, merge them
        if not (last_box[2] < current_box[0] or last_box[3] < current_box[1]):
            merged_box = (min(last_box[0], current_box[0]), min(last_box[1], current_box[1]), max(last_box[2], current_box[2]), max(last_box[3], current_box[3]))
            merged_mask = np.logical_or(last_mask, current_mask)

            merged_boxes[-1] = merged_box
            merged_masks[-1] = merged_mask
        else:
            # If the current box does not intersect with the last box, add it to the lists
            merged_boxes.append(current_box)
            merged_masks.append(current_mask)

    return merged_boxes, merged_masks
"""
def mask_size(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.vstack(contour.reshape(-1, 2) for contour in contours)
    print(points)
    distances_matrix = distance.pdist(points)
    print(distances_matrix)
    pos_max_dist = np.unravel_index(np.argmax(distances_matrix), (len(points), len(points)))
    pos_min_dist = np.unravel_index(np.argmin(distances_matrix), (len(points), len(points)))
    print(pos_max_dist)
    print(pos_min_dist)
    furthest_points = points[pos_max_dist]
    closest_points = points[pos_min_dist]

    return closest_points, furthest_points
"""
def dist(points):
    total_dist = 0
    for i in range(len(points) - 1):
        print(distance.euclidean(points[i], points[i+1]))
        total_dist += distance.euclidean(points[i], points[i+1])
    return total_dist

def mask_size(mask):
    skel = skeletonize(mask)
    skeleton = skel.astype(np.uint8)
    skel = skel.astype(np.uint8)*255
    #print(skel)
    points = np.argwhere(skel==255)
    #print(points)
    #distance = dist(points)
    distance = len(points)
    skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    #cv2.putText(skel, str(distance), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imwrite("./Fine_tuned_Detectron2/data/skeleton/{}_{}.jpg".format(j,i), skel)
    return skeleton, distance

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
    """
    print(img.shape)
    image = torch.tensor(img).permute(2, 0, 1).float()
    image = model.preprocess_image(image)
    print(image.shape)
    image = image.unsqueeze(0)
    #image.to(DEVICE)
    
    
    
    features = model.backbone(image)
    proposals, _ = model.proposal_generator(image, features, None)
    print(proposals)
    instances, _ = model.roi_heads(image, features, proposals, None)
    print(instances)
    mask_features = [features[f] for f in model.roi_heads.in_features]
    mask_features = model.roi_heads.mask_pooler(mask_features, [x.proposal_boxes for x in instances])
    print(mask_features)
    """
    
    
    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        outputs = model([inputs])
    #print(outputs)

    pred_masks = outputs[0]["instances"].pred_masks.cpu().numpy()
    #print(pred_masks)
    pred_boxes = outputs[0]["instances"].pred_boxes.tensor.cpu().numpy()

    if len(pred_masks) == 0:
        continue

    pred_boxes, pred_masks = merge_boxes_and_masks(pred_boxes, pred_masks)
    joined_masks = np.zeros_like(pred_masks[0])

    for i in range(len(pred_masks)):
        mask = pred_masks[i].astype(np.uint8)
        mask_grey = mask*255
        #print(mask)
        box = pred_boxes[i]

        #print(box)
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #print(x1, y1, x2, y2)
        w = x2 - x1
        h = y2 - y1
        skeleton, length = mask_size(mask)
        color = extract_color(img, skeleton)
        #print(h)
        #w = math.trunc((w*3500)/img.shape[1])
        #h = math.trunc((h*2625)/img.shape[0]) 
        text = "{}pp, {}".format(length,color)
        put_text(img, text, x1, y1)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        joined_masks = joined_masks + mask_grey

    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("./Fine_tuned_Detectron2/data/masks/{}.jpg".format(j), joined_masks)
    cv2.imwrite("./Fine_tuned_Detectron2/data/predictions/{}.jpg".format(j), img)
    #cv2.imwrite("./Fine_tuned_Detectron2/data/predictions/{}.jpg".format(j), out.get_image()[:, :, ::-1])
    #cv2.imshow("Image", out.get_image()[:, :, ::-1])
print("Done")
    