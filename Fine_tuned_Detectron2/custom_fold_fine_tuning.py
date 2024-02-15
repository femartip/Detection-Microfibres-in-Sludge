import json
import logging
import os
from collections import OrderedDict, defaultdict
import torch
import time
import datetime
from sklearn.model_selection import StratifiedKFold
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
from detectron2.engine import default_writers
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu" 
seed = 42
torch.manual_seed(seed)
NUM_FOLDS = 5

#https://github.com/cleanlab/examples/blob/f85155bb4e5d5643f878a2ccaa9363a1896619ce/object_detection/detectron2_training-kfold.ipynb#L45
def split_data(train_indices, test_indices, image_ids, image_data, data):
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    common_metadata = ['info', 'licenses', 'categories']

    for cm in common_metadata:
        train_data[cm] = data[cm]
        test_data[cm] = data[cm]

    train_image_ids = set([image_ids[i] for i in train_indices])
    test_image_ids = set([image_ids[i] for i in test_indices])

    for image in image_data:
        image_id = int(image['file_name'].split('.')[0])
        if image_id in train_image_ids:
            train_data['images'].append(image)
        else:
            test_data['images'].append(image)

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id in train_image_ids:
            train_data['annotations'].append(annotation)
        else:
            test_data['annotations'].append(annotation)

    return train_data, test_data

def print_data_info(data_dict, fold):
    images_count = len(data_dict['images'])
    annotations_count = len(data_dict['annotations'])
    print(f"Number of images: {images_count}, Number of annotations: {annotations_count}")

def k_fold_data(image_ids, category_ids, image_data, data):
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    pairs = []
    for fold, (train_indices, test_indices) in enumerate(skf.split(image_ids, category_ids)):
        print(f"Fold {fold} has {len(train_indices)} training data and {len(test_indices)} testing data")
        train_data, test_data = split_data(train_indices, test_indices, image_ids, image_data, data)
        train_file = f"./Fine_tuned_Detectron2/data/train_coco_{fold}_fold.json"
        test_file = f"./Fine_tuned_Detectron2/data/test_coco_{fold}_fold.json"
        with open(train_file, 'w') as train_file:
            json.dump(train_data, train_file)
        with open(test_file, 'w') as test_file:
            json.dump(test_data, test_file)
        print(f"Data info for training data fold {fold}:")
        print_data_info(train_data, fold)
        print(f"Data info for testing data fold {fold}:")
        print_data_info(test_data, fold)
        pairs.append([train_file, test_file])
    
    return pairs    

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


def do_test(cfg, model):
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
            #storage.put_scalar('bbox/AP', results_i['bbox']['AP'])
            #storage.put_scalar('segm/AP', results_i['segm']['AP'])
            #storage.put_scalar('bbox/AP50', results_i['bbox']['AP50'])
            #storage.put_scalar('segm/AP50', results_i['segm']['AP50'])
            #storage.put_scalar('bbox/AP75', results_i['bbox']['AP75'])
            #storage.put_scalar('segm/AP75', results_i['segm']['AP75'])
            #storage.put_scalar('bbox/APs', results_i['bbox']['APs'])
            #storage.put_scalar('segm/APs', results_i['segm']['APs'])
            #storage.put_scalar('bbox/APm', results_i['bbox']['APm'])
            #storage.put_scalar('segm/APm', results_i['segm']['APm'])
            #storage.put_scalar('bbox/APl', results_i['bbox']['APl'])
            #storage.put_scalar('segm/APl', results_i['segm']['APl'])
            
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume,model_name):
    model.train()   #Set model to training mode
    optimizer = build_optimizer(cfg, model)  #Build an optimizer from config
    scheduler = build_lr_scheduler(cfg, optimizer)  #Build a LR scheduler from config
    print(f"Scheduler: {scheduler}")
    print(f"Optimizer: {optimizer}")
    
    #If resuming from checkpoint 
    max_iter = cfg.SOLVER.MAX_ITER  #Max number of iterations

    #Create writers for logging
    writers = default_writers(cfg.OUTPUT_DIR, max_iter)

    #Checkpointer is used to save/load model
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    #checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    #print(f"Checkpointer: {checkpointer}")
    
    #Periodic checkpointer is used to save model periodically
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    #Creates start iter from checkpointer if so
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1)

    data_loader = build_detection_train_loader(cfg)     #Build a dataloader for object detection with some default features.
    print(data_loader)
    
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
            
            """
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model, storage) #Test model
                # Compared to "train_net.py", the test results are not dumped to EventStorage
            """
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
        #Save model
        checkpointer.save("model_final" + model_name)  


def setup(pairs,k):
    #Create config file
    cfg = get_cfg()
    train_dataset = []
    val_dataset = []
    for k in range(0, NUM_FOLDS):
        train = pairs[k][0].name
        validation = pairs[k][1].name
        register_coco_instances("train_" + str(k), {}, train, "./Fine_tuned_Detectron2/data/Dataset/images")
        register_coco_instances("validation_" + str(k), {}, validation, "./Fine_tuned_Detectron2/data/Dataset/images")
        train_dataset.append("train_" + str(k))
        val_dataset.append("validation_" + str(k))
    cfg.OUTPUT_DIR = "./Fine_tuned_Detectron2/models"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.TEST.EVAL_PERIOD = 50
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.MAX_ITER = 1000    # iterations to train for
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    return cfg



def main():
    data = json.load(open("./Fine_tuned_Detectron2/data/dataset.json", 'r'))    #Load coco format json
   
    image_data = data['images'] #Get image data
    annotations = data['annotations'] #Get annotations
    
    # Get unique image IDs and create a mapping of image ID to file name
    image_ids = [annotations[i]['image_id'] for i in range(len(annotations))]
    # Get category ID for each image
    category_ids = [annotations[i]['category_id'] for i in range(len(annotations))]

    pairs = k_fold_data(image_ids, category_ids, image_data, data) #Split data into k folds
    
    results_file = open("./Fine_tuned_Detectron2/models/results.txt", "a")

    batch_size = [3]
    lrates = [0.001]
    batch_size_per_image = [128,256,512]
    #batch_size = [1,2,3]
    #lrates = [0.001,0.0001,0.00001]
    #batch_size_per_image = [128,256,512]

    cfg = setup(pairs, NUM_FOLDS)
    max_result = {"bbox": {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0}, "segm": {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0}}
    max_result_n = 0
    count = 18
    
    for bs in batch_size:
        for lr in lrates:
            for bsi in batch_size_per_image:
                total_results = []
                cfg.SOLVER.IMS_PER_BATCH = bs
                cfg.SOLVER.BASE_LR = lr
                cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bsi
                print(f"Batch size: {bs}, Learning rate: {lr}, Batch size per image: {bsi}")
                results_file.write(str(count) + "\n")
                results_file.write("---------------------------------------------------------------------\n")
                results_file.write(f"Batch size: {bs}, Learning rate: {lr}, Batch size per image: {bsi}\n")
                for k in range(0, NUM_FOLDS):
                    cfg.DATASETS.TRAIN = ("train_" + str(k),)
                    cfg.DATASETS.TEST = ("validation_" + str(k),)
                    model = build_model(cfg)    #Build model
                    model.to(DEVICE)
                    do_train(cfg, model, False, str(count) + "_" + str(k))    #Train model
                    results = do_test(cfg, model)    #Test model
                    print(f"Fold {k} results: {results}")
                    results_file.write(f"Fold {k} results: {results}\n")
                    total_results.append(results)
                    config_yaml_path = "./Fine_tuned_Detectron2/models/config"+str(count) + "_" + str(k)+".yaml"
                    with open(config_yaml_path, 'w') as file:
                        yaml.dump(cfg, file)
                
                mean_total_results = {"bbox": {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0}, "segm": {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0}}
                for dict in total_results:
                    for key in dict:
                        for j in dict[key]:
                            mean_total_results[key][j] += dict[key][j]
                for key in mean_total_results:
                    for j in mean_total_results[key]:
                        mean_total_results[key][j] = mean_total_results[key][j] / NUM_FOLDS
                print(mean_total_results)
                if mean_total_results["bbox"]["AP"] > max_result["bbox"]["AP"] and mean_total_results["bbox"]["AP"] > max_result["bbox"]["AP"]:
                    max_result = mean_total_results
                    max_result_n = count
                results_file.write(f"Mean performance of {NUM_FOLDS} folds: {mean_total_results} \n")
                count += 1
    results_file.write(f"Best results {max_result_n} with {mean_total_results}")
    results_file.close()

"""
    for k in range(0, NUM_FOLDS):
        #result_dict = {}
        train_data = pairs[k][0].name
        val_data = pairs[k][1].name
        if not trained_model:
            cfg = setup(train_data, val_data,k) #Create config file
        elif trained_model:
            cfg = setup(cfg, train_data, val_data,k)
        model = build_model(cfg)    #Build model
        model.to(DEVICE)
        do_train(cfg, model, trained_model)    #Train model
        if not trained_model:
            trained_model = True
"""
        

if __name__ == "__main__":
    main()