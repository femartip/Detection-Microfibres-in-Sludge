import logging
import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Utils.cocosplit_cross_val import k_fold_data
import cv2
import numpy as np
import argparse
#from sklearn.metrics import average_precision_score
from torchmetrics.functional.classification import average_precision
from unet import UNet
from unet import BB_Unet
from pycocotools.coco import COCO
import json

from SegmentationDataset import get_k_fold_dataset, CocoMaskDataset

SEED = 42
FOLD = 0

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), targets

def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: torch.Tensor, target: torch.Tensor):
    return 1 - dice_coeff(input, target)

def calculate_mAP(preds, targets, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) for binary segmentation.

    Arguments:
        preds: Tensor of predicted masks (B, H, W).
        targets: Tensor of ground truth masks (B, H, W).
        threshold: Threshold to binarize predictions.

    Returns:
        mean Average Precision (mAP) score.
    """
    # Apply threshold to predictions
    #preds_bin = (preds > threshold).float()
    
    #preds_bin_flatten = torch.flatten(preds_bin)
    #targets_flatten = torch.flatten(targets)
    ap = average_precision(preds, targets.long(), task="binary")
    return ap

def calculate_accuracy(preds, targets):
    preds_bin = (preds > 0.5).float()
    correct = (preds_bin == targets).float()  
    accuracy = correct.mean().item()  
    return accuracy

def evaluate_model(model, val_loader, threshold=0.5):
    model.eval()
    aps = []
    accuracies = []

    with torch.no_grad():
        for images, true_masks in val_loader:
            masks_pred = model(images).squeeze(1)
            masks_pred = torch.sigmoid(masks_pred)

            ap = calculate_mAP(masks_pred, true_masks, threshold)
            accuracy = calculate_accuracy(masks_pred, true_masks)

            aps.append(ap)
            accuracies.append(accuracy)

    mAP = sum(aps) / len(aps)
    mean_accuracy = sum(accuracies) / len(accuracies)
    return mAP, mean_accuracy

def train_model(model,device, train_dataset, val_dataset, epochs=100, learning_rate=0.001, batch_size=4):
    weight_decay = 0.000000001
    #momentum = 0.999
    #gradient_clipping = 1.0

    logging.info(f'''Starting training: Epochs:{epochs} Batch size:{batch_size} Learning rate:{learning_rate} Device:{device.type}''')

    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10) 
    
    loss_bce = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    metrics = {}

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        model.to(device=device)
        model.train(True)
        num_batches = 0
        losses = []
        accuracies = []
        aps = []

        for idx, batch in enumerate(train_loader):
            images, true_masks = batch
            images = images.to(device=device)
            true_masks = torch.stack(true_masks).to(device=device)
            logging.debug(f"Batch {idx}, images: {images.shape}, masks: {true_masks.shape}")

            masks_pred = model(images)  # Forward pass
            masks_pred = masks_pred.squeeze(1)
            logging.debug(f"predicted masks: {masks_pred.shape}")
            logging.debug(f"Max value pred: {torch.sigmoid(masks_pred).max()}, Max value true {true_masks.max()}")
            #loss = combined_loss(masks_pred, true_masks, metrics)    # Calculates the loss as combination of BCE and Dice loss, this ensures pixel level precision
            loss = loss_bce(masks_pred, true_masks) 
            loss += dice_loss(torch.sigmoid(masks_pred), true_masks)
            optimizer.zero_grad()   # Zero gradients
            
            loss.backward() 

            optimizer.step()

            logging.debug(f"Loss: {loss.item()}")

            accuracy = calculate_accuracy(torch.sigmoid(masks_pred), true_masks)
            logging.debug(f"Accuracy: {accuracy}")
            accuracies.append(accuracy)

            try:
                ap = calculate_mAP(torch.sigmoid(masks_pred), true_masks)
            except Exception as e:
                logging.error(f"Error calculating mAP: {e}")
                logging.error(f"Predictions: {torch.min(masks_pred)}, {torch.max(masks_pred)}")
                logging.error(f"True masks: {torch.min(true_masks)}, {torch.max(true_masks)}")
                ap = 0

            logging.debug(f"mAP: {ap}")
            aps.append(ap)
            losses.append(loss.item())
            
            if num_batches % 100 == 0:
                print(f"Batch: {num_batches}, Loss: {loss.item()}, Accuracy: {accuracy}, mAP: {ap}")

            num_batches += 1
            
        #scheduler.step(sum(aps) / num_batches)

        avg_epoch_loss = sum(losses) / num_batches
        avg_epoch_acc = sum(accuracies) / num_batches
        mean_ap = sum(aps) / num_batches
        print(f"Training Loss: {avg_epoch_loss:.4f}")
        print(f"Training Accuracy: {avg_epoch_acc:.4f}")
        print(f"Training mAP: {mean_ap:.4f}")

        
        metrics[epoch+1] = {"train_loss": avg_epoch_loss, "train_accuracy": avg_epoch_acc, "train_mAP": mean_ap}
        
        # model.to("cpu")
        #mAP_five, mean_accuracy_five = evaluate_model(model, val_loader, threshold=0.5)
        #model.to(device)
        #val_losses.append(mAP_five)
        #val_accuracies.append(mean_accuracy_five)
        #print(f"Validation mAP |IoU 0.5:0.95|: {mAP_five:.4f}, Validation Accuracy: {mean_accuracy_five:.4f}")

    #np.save("UNet/results/train_losses_fold_{}.npy".format(FOLD), train_losses)
    #np.save("UNet/results/train_accuracies_fold_{}.npy".format(FOLD), train_accuracies)
    #np.save("UNet/results/val_losses_fold_{}.npy".format(FOLD), val_losses)
    #np.save("UNet/results/val_accuracies_fold_{}.npy".format(FOLD), val_accuracies)
    
    #torch.save(model.state_dict(), "UNet/models/model_fold_{}.pth".format(FOLD))
    return metrics


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="./UNet/data/Dataset/Dataset_vidrio")
    args.add_argument("--device", type=str, default="cuda:1")
    args.add_argument("--model", type=str, default="UNet")
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--learning_rate", type=float, default=0.001)

    args = args.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_dir = args.data_dir

    if args.model == "UNet":
        model = UNet(in_channels=3, num_classes=1)
    elif args.model == "BB_Unet":
        model = BB_Unet()
    else:
        raise ValueError("Model not supported")

    NUM_FOLDS = 5
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Using device: ", device)
    if device.type != 'cpu':
        torch.cuda.empty_cache()
    
    results = {}

    for fold in range(NUM_FOLDS):
        print("Segmentation fold: ", fold)
        FOLD = fold
        train_coco = COCO(os.path.join(data_dir, f"train_coco_{fold}_fold.json"))
        test = COCO(os.path.join(data_dir, f"test_coco_{fold}_fold.json"))
        print("Train ids: {}, annotations: {}".format(len(train_coco.getImgIds()), len(train_coco.getAnnIds(train_coco.getImgIds()))))
        print("Test ids: {}, annotations: {}".format(len(test.getImgIds()), len(test.getAnnIds(test.getImgIds()))))
        
        train_dataset = CocoMaskDataset(os.path.join(data_dir, "images"), os.path.join(data_dir, f"train_coco_{fold}_fold.json"))
        
        val_dataset = CocoMaskDataset(os.path.join(data_dir, "images"), os.path.join(data_dir, f"test_coco_{fold}_fold.json"))
        result = train_model(model, device, train_dataset, val_dataset, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size)
        results[fold] = result

    print("Final results:")
    print("Mean train loss: ", np.mean([results[fold][epoch]["train_loss"] for fold in results for epoch in results[fold]]))
    print("Mean train accuracy: ", np.mean([results[fold][epoch]["train_accuracy"] for fold in results for epoch in results[fold]]))
    print("Mean train mAP: ", np.mean([results[fold][epoch]["train_mAP"] for fold in results for epoch in results[fold]]))
        