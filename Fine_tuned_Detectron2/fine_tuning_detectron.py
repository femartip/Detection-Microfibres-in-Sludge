import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import torch
import glob
import json
from PIL import Image
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
#import torchmetrics
import pycocotools.mask as mask_util

import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
#from torch.utils.tensorboard import SummaryWriter
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import DatasetEvaluator, COCOEvaluator

import yaml

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
      return COCOEvaluator(dataset_name, distributed=False ,output_dir=output_folder, use_fast_impl=False)


def train():
  #writer = SummaryWriter()
  register_coco_instances("dataset_train", {}, "./Fine_tuned_Detectron2/data/train.json", "./Fine_tuned_Detectron2/data/Dataset/images")
  register_coco_instances("dataset_val", {}, "./Fine_tuned_Detectron2/data/val.json", "./Fine_tuned_Detectron2/data/Dataset/images")
  train_metadata = MetadataCatalog.get("dataset_train")
  dataset_dicts = DatasetCatalog.get("dataset_train")

  cfg = get_cfg()
  cfg.OUTPUT_DIR = "./Fine_tuned_Detectron2/models"
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TRAIN = ("dataset_train",)
  cfg.DATASETS.TEST = ("dataset_val",)
  cfg.TEST.EVAL_PERIOD = 50
  cfg.DATALOADER.NUM_WORKERS = 0
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
  cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
  cfg.SOLVER.STEPS = []        # do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # Default is 512, using 256 for this dataset.
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
  #cfg.INPUT.MASK_FORMAT='polygon'
  #NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = MyTrainer(cfg) #Create an instance of of DefaultTrainer with the given congiguration
  #trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=False)
  
  trainer.train()
  #writer.flush()

  config_yaml_path = "./Fine_tuned_Detectron2/models/config.yaml"
  with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)

  #print(trainer)
  #losses = trainer.storage.history["total_loss"]
  #plt.plot(losses)
  #plt.xlabel('Iteration')
  #plt.ylabel('Loss')
  #plt.savefig("./Fine_tuned_Detectron2/loss.png")

if __name__ == "__main__":
  train()