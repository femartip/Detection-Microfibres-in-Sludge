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
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification.accuracy import Accuracy

from unet import UNet

SEED = 42

def get_dataset(data_dir: str, num_folds: int = 5):    
    k_fold_dict =  k_fold_data(data_dir, NUM_FOLDS=num_folds, seed=SEED)
    return k_fold_dict  #{0: {"train": ['info', 'licenses', 'categories', 'images', 'annotations'], "test": ['info', 'licenses', 'categories', 'images', 'annotations']}}


def train_model(model,device, X_train, y_train, X_val, y_val):
    epochs = 5
    batch_size = 1
    learning_rate = 1e-5
    val_percent = 0.1
    save_checkpoint = True
    img_scale = 0.5
    amp = False
    weight_decay = 1e-8
    momentum = 0.999
    gradient_clipping = 1.0

    logging.info(f'''Starting training: Epochs:{epochs} Batch size:{batch_size} Learning rate:{learning_rate} Checkpoints:{save_checkpoint} Device:{device.type} Images scaling:{img_scale} Mixed Precision:{amp}''')

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) # Automatic Mixed Precision, increases speed and reduces memory usage
    criterion =  nn.BCEWithLogitsLoss()
    AP_metric = MeanAveragePrecision(num_classes=1, iou_type='segm', extended_summary=True)
    #accuracy_metric = Accuracy()
    model.train()
    global_step = 0
    epoch_loss = 0

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        for idx, batch in enumerate(train_loader):
            images, true_masks = batch
            images = images.clone().detach().float()
            images = images.to(device=device)
            true_masks = true_masks.to(device=device)

            
            masks_pred = model(images)
            masks_pred = masks_pred[0]
            loss = criterion(masks_pred.squeeze(1), true_masks.float())

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            AP_metric.update(masks_pred, true_masks)
            #accuracy_metric(masks_pred, true_masks)

            logging.info(f'Epoch: {epoch + 1} Loss: {loss.item()}')
            logging.info(f"Metrics: AP IoU=0.5:0.95 | area = all: {AP_metric.compute()} \n")
            # Evaluation round
            """
            division_step = (idx // (5 * batch_size))
            if division_step > 0:
                if global_step % division_step == 0:
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)
                    logging.info('Validation Dice score: {}'.format(val_score))
                    """

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model.to(device=device)
    data_dir = "./Fine_tuned_Detectron2/data/Dataset/Dataset_vidrio"
    data = get_dataset(data_dir)
    
    results = {}

    for fold in data:
        X_train = data[fold]['train']['images']  
        X_train = [cv2.cvtColor(cv2.imread(os.path.join(data_dir, "images/" + img['file_name'])), cv2.COLOR_BGR2RGB) for img in X_train]
        X_train = np.array([np.transpose(img, (2, 0, 1)) for img in X_train])
        y_train = data[fold]['train']['annotations']
        X_test = data[fold]['test']['images']
        X_test = [cv2.cvtColor(cv2.imread(os.path.join(data_dir, "images/" + img['file_name'])), cv2.COLOR_BGR2RGB) for img in X_test]
        y_test = data[fold]['test']['annotations']

        train_model(model, device, X_train, y_train, X_test, y_test)
        