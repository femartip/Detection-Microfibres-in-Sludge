import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from Utils.cocosplit_cross_val import k_fold_data
import cv2
import numpy as np
from sklearn.metrics import average_precision_score
from unet import UNet
from pycocotools.coco import COCO
import json

from SegmentationDataset import get_k_fold_dataset, SegmentationDataset

SEED = 42
FOLD = 0

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
    preds = (preds > threshold).float()
    
    # Flatten predictions and targets for computing AP
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()

    # Calculate average precision score
    ap = average_precision_score(targets, preds)
    return ap

def calculate_accuracy(preds, targets):
    correct = (preds == targets).float()  
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

def train_model(model,device, train_dataset, val_dataset):
    epochs = 5
    batch_size = 8
    learning_rate = 0.0001
    val_percent = 0.1
    save_checkpoint = False
    img_scale = 0.5
    if device.type == 'cuda':
        amp = True
    else:
        amp = False
    weight_decay = 0.000000001
    momentum = 0.999
    gradient_clipping = 1.0

    logging.info(f'''Starting training: Epochs:{epochs} Batch size:{batch_size} Learning rate:{learning_rate} Checkpoints:{save_checkpoint} Device:{device.type} Images scaling:{img_scale} Mixed Precision:{amp}''')

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) # Automatic Mixed Precision, increases speed and reduces memory usage
    criterion =  nn.BCEWithLogitsLoss()
    
    global_step = 0
    epoch_loss = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(epochs):
        model.train()
        model.to(device=device)
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for idx, batch in enumerate(train_loader):
            images, true_masks = batch
            images = images.to(device=device)
            true_masks = true_masks.to(device=device)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = model.forward(images)
                loss = criterion(masks_pred.squeeze(1), true_masks)
                #loss = loss.detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            #grad_scaler.scale(loss).backward()
            #grad_scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            #grad_scaler.step(optimizer)
            #grad_scaler.update()
        
            batch_accuracy = calculate_accuracy(torch.sigmoid(masks_pred), true_masks)
            epoch_acc += batch_accuracy
            epoch_loss += loss.item()
            num_batches += 1

            
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_acc = epoch_acc / num_batches
        print(f"Epoch: {epoch + 1}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_acc:.4f}")
        model.to("cpu")
        mAP_five, mean_accuracy_five = evaluate_model(model, val_loader, threshold=0.5)
        model.to(device)
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(avg_epoch_acc)
        val_losses.append(mAP_five)
        val_accuracies.append(mean_accuracy_five)
        print(f"Validation mAP |IoU 0.5:0.95|: {mAP_five:.4f}, Validation Accuracy: {mean_accuracy_five:.4f}")

        np.save("UNet/results/train_losses_fold_{}.npy".format(FOLD), train_losses)
        np.save("UNet/results/train_accuracies_fold_{}.npy".format(FOLD), train_accuracies)
        np.save("UNet/results/val_losses_fold_{}.npy".format(FOLD), val_losses)
        np.save("UNet/results/val_accuracies_fold_{}.npy".format(FOLD), val_accuracies)
        
        torch.save(model.state_dict(), "UNet/models/model_fold_{}.pth".format(FOLD))


if __name__ == '__main__':
    NUM_FOLDS = 5
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logging.info(f'Using {torch.cuda.get_device_name(0)}')

    model = UNet(n_classes=1)
    
    data_dir = "./UNet/data/Dataset/Dataset_vidrio"
    
    results = {}

    for fold in range(NUM_FOLDS):
        print("Segmentation fold: ", fold)
        FOLD = fold
        train_coco = COCO(os.path.join(data_dir, f"train_coco_{fold}_fold.json"))
        test = COCO(os.path.join(data_dir, f"test_coco_{fold}_fold.json"))
        print("Train ids: {}, annotations: {}".format(len(train_coco.getImgIds()), len(train_coco.getAnnIds(train_coco.getImgIds()))))
        print("Test ids: {}, annotations: {}".format(len(test.getImgIds()), len(test.getAnnIds(test.getImgIds()))))
        
        train_dataset = SegmentationDataset(COCO(data_dir + "/coco_format.json"), os.path.join(data_dir, "images"))
        val_dataset = SegmentationDataset(COCO(data_dir + "/coco_format.json"), os.path.join(data_dir, "images"))
        train_model(model, device, train_dataset, val_dataset)
        