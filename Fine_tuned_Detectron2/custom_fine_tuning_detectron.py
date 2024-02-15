import logging
import os
from collections import OrderedDict, defaultdict
import torch
import time
import datetime
import yaml

from fvcore.common.timer import Timer

from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_writers, BestCheckpointer
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data.datasets import register_coco_instances


logger = logging.getLogger("detectron2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu" 
seed = 42
torch.manual_seed(seed)

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR)
    #print(output_folder)
    if evaluator_type == "coco":
        #evaluator = COCOEvaluator(dataset_name, tasks=("segm"), distributed=False ,output_dir=output_folder, use_fast_impl=False)
        evaluator = COCOEvaluator(dataset_name, distributed=False ,output_dir=output_folder, use_fast_impl=False)
    else :  
        raise NotImplementedError("Unsupported dataset type {} for evaluation.".format(evaluator_type))
    
    return evaluator


def do_test(cfg, model, storage):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        #print(type(dataset_name))
        data_loader = build_detection_test_loader(cfg, dataset_name)
        #print(data_loader)
        # Create evaluator
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator) #Sets model.eval temporarily
        #print(results_i)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            storage.put_scalar('bbox/AP', results_i['bbox']['AP'])
            storage.put_scalar('segm/AP', results_i['segm']['AP'])
            storage.put_scalar('bbox/AP50', results_i['bbox']['AP50'])
            storage.put_scalar('segm/AP50', results_i['segm']['AP50'])
            storage.put_scalar('bbox/AP75', results_i['bbox']['AP75'])
            storage.put_scalar('segm/AP75', results_i['segm']['AP75'])
            storage.put_scalar('bbox/APs', results_i['bbox']['APs'])
            storage.put_scalar('segm/APs', results_i['segm']['APs'])
            storage.put_scalar('bbox/APm', results_i['bbox']['APm'])
            storage.put_scalar('segm/APm', results_i['segm']['APm'])
            storage.put_scalar('bbox/APl', results_i['bbox']['APl'])
            storage.put_scalar('segm/APl', results_i['segm']['APl'])
            
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()   #Set model to training mode
    optimizer = build_optimizer(cfg, model)  #Build an optimizer from config
    scheduler = build_lr_scheduler(cfg, optimizer)  #Build a LR scheduler from config
    print(f"Scheduler: {scheduler}")
    print(f"Optimizer: {optimizer}")

    #Checkpointer is used to save/load model
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    
    #Creates start iter from checkpointer if so
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER  #Max number of iterations

    #Periodic checkpointer is used to save model periodically
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    #Create writers for logging
    writers = default_writers(cfg.OUTPUT_DIR, max_iter)

    data_loader = build_detection_train_loader(cfg)     #Build a dataloader for object detection with some default features.
    
    #print(type(data_loader))
    print("Starting training from iteration {}".format(start_iter))

    logger.info("Starting training from iteration {}".format(start_iter)) #Start logging
    with EventStorage(start_iter) as storage: #Store events
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):   #Iterate over data
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()

            storage.iter = iteration    #Set storage iteration
            
            #print(len(data))   #Prints number of images in batch
            #print(data)    #Data contains JSON data for each image in batch
            
            loss_dict = model(data) #Get loss dict, this inclueds losses for cls, box_reg, mask, rpn
            #print(loss_dict)
            losses = sum(loss_dict.values())    #Sum losses
            assert torch.isfinite(losses).all(), loss_dict  #Check if losses are finite

            #metrics_dict = loss_dict
            #metrics_dict["data_time"] = 

            optimizer.zero_grad()   #Zero gradients
            losses.backward()   #Backpropagate
            optimizer.step()    #Update weights
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False) #Log learning rate
            
            storage.put_scalars(total_loss=losses)
            storage.put_scalars(**loss_dict)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()    #Step scheduler

            print("Iteration {}: {}".format(iteration, losses))
            #Every test period evaluate the model
            
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model, storage) #Test model
                # Compared to "train_net.py", the test results are not dumped to EventStorage
            
            #Evaluation every 20 iterations
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                #Log metrics with writer
                for writer in writers:
                    writer.write()

                
                

            periodic_checkpointer.step(iteration) #Step periodic checkpointer
            

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(str(datetime.timedelta(seconds=int(total_time)))))

def setup():
    #Create config file
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "./Fine_tuned_Detectron2/models"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ("dataset_val",)
    cfg.TEST.EVAL_PERIOD = 50
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 3  # Batch size
    cfg.SOLVER.BASE_LR = 0.001  # LR
    cfg.SOLVER.MAX_ITER = 1000    # iterations to train for
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # Default is 512, using 256 for this dataset.
    cfg.freeze()
    return cfg


def main():
    register_coco_instances("dataset_train", {}, "./Fine_tuned_Detectron2/data/train.json", "./Fine_tuned_Detectron2/data/Dataset/images")
    register_coco_instances("dataset_val", {}, "./Fine_tuned_Detectron2/data/val.json", "./Fine_tuned_Detectron2/data/Dataset/images")
    cfg = setup() #Create config file

    model = build_model(cfg)    #Build model
    model.to(DEVICE)
   
    do_train(cfg, model)   #Train model
    
    #Save cfg
    config_yaml_path = "./Fine_tuned_Detectron2/models/config.yaml"
    with open(config_yaml_path, 'w') as file:
        yaml.dump(cfg, file)


if __name__ == "__main__":
    main()