import torch
import os
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor, sam_model_registry
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
import json
import io
import contextlib

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
FILTER = "CA"
# FILTER = "glass"

def get_model():
    sam = sam_model_registry[MODEL_TYPE](checkpoint="./models/sam_vit_b_01ec64.pth")
    sam.to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def mask_to_RLE(mask):
    rle = mask_util.encode(mask.astype(np.uint8, order='F'))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle
    

def import_images(path):
    images = []
    count = 0
    print("Importing images...")
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            #print("Processing file: {}".format(file))
            image_bgr = cv2.imread(os.path.join(path, file))
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            images.append((file, img))
            if count > 4:
                break
            count += 1
    return images

def detect_fibres(images, ground_truth_path):
    predictions = []
    mask_generator = get_model()
    
    ground_truth_anns = load_labels(ground_truth_path)
    label_images = ground_truth_anns['images']
    
    print("Detecting fibres...")
    for idx, (file_name, img) in enumerate(images):
        masks = mask_generator.generate(img)
    
        label_image = [x for x in label_images if x['file_name'] == file_name][0]
        image_id = label_image['id']

        for mask in masks:
            segmentation = mask_to_RLE(mask['segmentation'])
            predictions.append({
                "image_id": image_id,
                "file_name": file_name,
                "bbox": mask['bbox'],
                "segmentation": segmentation,
                "score": mask['stability_score']  # SAM provides scores for each mask
            })
    return predictions

# Coco result format https://cocodataset.org/#format-results
def save_coco_json(predictions, output_path, images):
    coco_results = []

    for idx, pred in enumerate(predictions):
        coco_result_format = {
            "image_id": pred["image_id"],
            "category_id": 1,
            "segmentation": pred["segmentation"],   # polygon format
            "score": pred["score"],
        }
        coco_results.append(coco_result_format)
    

    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=4)

def load_labels(json_path):
    with open(json_path, 'r') as f:
        labels = json.load(f)
    return labels

def evaluate(predictions_path, ground_truth_path):
    coco_gt = COCO(ground_truth_path)
    print(coco_gt)
    coco_dt = coco_gt.loadRes(predictions_path)
    # Capture output for segmentation evaluation
    with io.StringIO() as segm_buf, contextlib.redirect_stdout(segm_buf):
        coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval_segm.evaluate()
        coco_eval_segm.accumulate()
        coco_eval_segm.summarize()
        segm_output = segm_buf.getvalue()
    
    # Capture output for bounding box evaluation
    with io.StringIO() as bbox_buf, contextlib.redirect_stdout(bbox_buf):
        coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()
        bbox_output = bbox_buf.getvalue()

    # Write results to file
    with open("./baselines/Zero_shot_SAM/output/{}_results.txt".format(FILTER), 'w') as f:
        f.write('COCO Evaluation Results\n\n')
        f.write('Segmentation Evaluation:\n')
        f.write(segm_output)
        f.write('\n')
        f.write('Bounding Box Evaluation:\n')
        f.write(bbox_output)


if __name__ == "__main__":
    if FILTER == "glass":
        filter_path = "vidrio"
    elif FILTER == "CA":
        filter_path = "CA"

    images = import_images('./Fine_tuned_Detectron2/data/Dataset/Dataset_{}/images/'.format(filter_path))
    predictions = detect_fibres(images, ground_truth_path='./Fine_tuned_Detectron2/data/Dataset/Dataset_{}/coco_format.json'.format(filter_path))
    save_coco_json(predictions, './baselines/Zero_shot_SAM/output/{}_coco_predictions.json'.format(FILTER), images)

    # Assuming you have ground truth labels in 'ground_truth.json'
    evaluate(predictions_path='./baselines/Zero_shot_SAM/output/{}_coco_predictions.json'.format(FILTER), ground_truth_path='./Fine_tuned_Detectron2/data/Dataset/Dataset_{}/coco_format.json'.format(filter_path))
