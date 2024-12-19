import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLossModified, compute_metrics, compute_pixel_accuracy, plot_metrics
from torchvision import transforms
from torch.nn.functional import one_hot
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

### Validation Function ###
def validate(model, val_loader, criterion, num_classes, device='cuda'):
    model.eval()
    total_loss = 0.0

    dice_scores = []
    iou_scores = []
    pixel_accuracies = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    specificity_scores = []

    all_probs = []
    all_gts = []

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = model(image_batch)
            loss = criterion(outputs, label_batch)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:,1,:,:]
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            all_probs.append(probs.detach().cpu().flatten())
            all_gts.append(label_batch.detach().cpu().flatten())

            pixel_accuracies.append(compute_pixel_accuracy(pred, label_batch))
            metrics = compute_metrics(pred, label_batch)
            dice_scores.append(metrics['dice'])
            iou_scores.append(metrics['iou'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
            specificity_scores.append(metrics['specificity'])

    avg_loss = total_loss / len(val_loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_pixel_acc = sum(pixel_accuracies) / len(pixel_accuracies)

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_specificity = sum(specificity_scores) / len(specificity_scores)

    all_probs = torch.cat(all_probs).numpy()
    all_gts = torch.cat(all_gts).numpy()
    try:
        avg_auc = roc_auc_score(all_gts, all_probs)
    except ValueError:
        avg_auc = float('nan')

    return avg_loss, avg_dice, avg_iou, avg_pixel_acc, avg_precision, avg_recall, avg_f1, avg_specificity, avg_auc

### TransUNet Trainer ###
def trainer_mrt1(args, model, snapshot_path):
    from datasets.dataset import MRT1_dataset, RandomGenerator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_metrics = {
        'loss': [], 'acc': [], 'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': [], 'auc': [], 'lr': []
    }
    val_metrics = {
        'loss': [], 'acc': [], 'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': [], 'auc': []
    }

    logging.basicConfig(filename=os.path.join(snapshot_path,"log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = MRT1_dataset(
        base_dir=args.train_root_path, 
        list_dir=args.list_dir, 
        split=args.train_split,
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    )
    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    print("The length of train set is: {}".format(len(db_train)))

    db_val = MRT1_dataset(
        base_dir=args.val_root_path, 
        list_dir=args.list_dir, 
        split=args.val_split,
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    )
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("The length of validation set is: {}".format(len(db_val)))



    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLossModified(n_classes=num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    iter_num = 0

    for epoch_num in iterator:
        model.train()
        train_loss_epoch = []
        train_acc_epoch = []
        train_dice_epoch = []
        train_iou_epoch = []
        train_precision_epoch = []
        train_recall_epoch = []
        train_f1_epoch = []
        train_specificity_epoch = []
        train_auc_epoch = []

        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            label_batch[label_batch > 0] = 1
            label_batch[label_batch == 0] = 0

            outputs = model(image_batch)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.2 * loss_ce + 0.8 * loss_dice

            pixel_accuracy = compute_pixel_accuracy(pred, label_batch)
            metrics = compute_metrics(pred, label_batch)
            dice_score = metrics['dice']
            iou_score = metrics['iou']
            precision_score = metrics['precision']
            recall_score = metrics['recall']
            f1_score = metrics['f1']
            specificity_score = metrics['specificity']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            train_loss_epoch.append(loss.item())
            train_acc_epoch.append(pixel_accuracy)
            train_dice_epoch.append(dice_score)
            train_iou_epoch.append(iou_score)
            train_precision_epoch.append(precision_score)
            train_recall_epoch.append(recall_score)
            train_f1_epoch.append(f1_score)
            train_specificity_epoch.append(specificity_score)

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/dice_score', dice_score, iter_num)
            writer.add_scalar('info/iou_score', iou_score, iter_num)
            writer.add_scalar('info/pixel_accuracy', pixel_accuracy, iter_num)
            writer.add_scalar('info/precision', precision_score, iter_num)
            writer.add_scalar('info/recall', recall_score, iter_num)
            writer.add_scalar('info/f1', f1_score, iter_num)

            logging.info(f'Epoch {epoch_num}, Iteration {iter_num}: Loss: {loss.item():.4f}, Dice: {dice_score:.4f}, IoU: {iou_score:.4f}, Acc: {pixel_accuracy:.4f}, Precision: {precision_score:.4f}, Recall: {recall_score:.4f}, F1: {f1_score:.4f}, Spec: {specificity_score:.4f}')

        train_metrics['loss'].append(np.mean(train_loss_epoch))
        train_metrics['acc'].append(np.mean(train_acc_epoch))
        train_metrics['dice'].append(np.mean(train_dice_epoch))
        train_metrics['iou'].append(np.mean(train_iou_epoch))
        train_metrics['precision'].append(np.mean(train_precision_epoch))
        train_metrics['recall'].append(np.mean(train_recall_epoch))
        train_metrics['f1'].append(np.mean(train_f1_epoch))
        train_metrics['specificity'].append(np.mean(train_specificity_epoch))
        train_metrics['auc'].append(float('nan'))
        train_metrics['lr'].append(lr_)

        val_loss, val_dice, val_iou, val_acc, val_precision, val_recall, val_f1, val_spec, val_auc = validate(model, val_loader, dice_loss, num_classes, device=device)

        logging.info("Validation Results - Epoch: {}  Loss: {:.4f} | Dice: {:.4f} | IoU: {:.4f} | Acc: {:.4f}".format(epoch_num, val_loss, val_dice, val_iou, val_acc))
        logging.info("Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} | Spec: {:.4f} | AUC: {:.4f}".format(val_precision, val_recall, val_f1, val_spec, val_auc))

        writer.add_scalar('val/loss', val_loss, epoch_num)
        writer.add_scalar('val/dice', val_dice, epoch_num)
        writer.add_scalar('val/iou', val_iou, epoch_num)
        writer.add_scalar('val/acc', val_acc, epoch_num)
        writer.add_scalar('val/precision', val_precision, epoch_num)
        writer.add_scalar('val/recall', val_recall, epoch_num)
        writer.add_scalar('val/f1', val_f1, epoch_num)
        writer.add_scalar('val/specificity', val_spec, epoch_num)
        if not np.isnan(val_auc):
            writer.add_scalar('val/auc', val_auc, epoch_num)

        val_metrics['loss'].append(val_loss)
        val_metrics['acc'].append(val_acc)
        val_metrics['dice'].append(val_dice)
        val_metrics['iou'].append(val_iou)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['f1'].append(val_f1)
        val_metrics['specificity'].append(val_spec)
        val_metrics['auc'].append(val_auc)

        if (epoch_num+1) % 10 == 0:
            plot_metrics(epoch_num+1, train_metrics, val_metrics, snapshot_path)

        save_interval = 25
        if epoch_num % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"save model to {save_mode_path}")

    writer.close()
    return "Training Finished!"
