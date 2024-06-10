import torch
import os
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
import json
import io
import contextlib

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model():
    cfg_inst = get_cfg()
    cfg_inst.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_inst.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo.  https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
    cfg_inst.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    model = DefaultPredictor(cfg_inst)

    return model

def mask_to_RLE(mask):
    rle = mask_util.encode(mask.astype(np.uint8, order='F'))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle
    

def import_images(path):
    images = []
    print("Importing images...")
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            #print("Processing file: {}".format(file))
            image_bgr = cv2.imread(os.path.join(path, file))
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            images.append((file, img))
    return images

def detect_fibres(images, ground_truth_path):
    predictions = []
    model = get_model()
    
    ground_truth_anns = load_labels(ground_truth_path)
    label_images = ground_truth_anns['images']
    
    print("Detecting fibres...")
    for idx, (file_name, img) in enumerate(images):
        outputs = model(img)
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        score = instances.scores.numpy()
        label_image = [x for x in label_images if x['file_name'] == file_name][0]
        image_id = label_image['id']

        for i in range(len(masks)):
            mask = masks[i].astype(np.uint8)
            x1, y1, x2, y2 = boxes[i]
            rle_mask = mask_to_RLE(mask)
            predictions.append({
                "image_id": image_id,
                "file_name": file_name,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": rle_mask,
                "score": float(score[i]) # float64 is not json serializable
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
    with open("./baselines/Zero_shot_Detectron2/output/ca_results.txt", 'w') as f:
        f.write('COCO Evaluation Results\n\n')
        f.write('Segmentation Evaluation:\n')
        f.write(segm_output)
        f.write('\n')
        f.write('Bounding Box Evaluation:\n')
        f.write(bbox_output)


if __name__ == "__main__":
    images = import_images('./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/images/')
    predictions = detect_fibres(images, ground_truth_path='./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/coco_format.json')
    save_coco_json(predictions, './baselines/Zero_shot_Detectron2/output/coco_predictions.json', images)

    # Assuming you have ground truth labels in 'ground_truth.json'
    evaluate(predictions_path='./baselines/Zero_shot_Detectron2/output/coco_predictions.json', ground_truth_path='./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/coco_format.json')