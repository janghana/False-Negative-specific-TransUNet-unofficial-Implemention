import os
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
import SimpleITK as sitk
import matplotlib.pyplot as plt

### Dice Loss Classes ###
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        return nn.functional.one_hot(input_tensor.long(), num_classes=self.n_classes).permute(0,3,1,2).float()

    def _dice_loss(self, score, target):
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f"Predict {inputs.size()} & target {target.size()} do not match"
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

class DiceLossModified(DiceLoss):
    def __init__(self, n_classes):
        super(DiceLossModified, self).__init__(n_classes)

    def _dice_loss(self, score, target, alpha=2.0, beta=1.5):
        smooth = 1e-5
        tp = torch.sum(score * target.float())
        fp = torch.sum(score * (1 - target.float()))
        fn = torch.sum((1 - score) * target.float())

        loss = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1 - loss

### Metric Functions ###
def safe_divide(numerator, denominator, epsilon=1e-5):
    return numerator / (denominator + epsilon)

def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {name}")
    if torch.isinf(tensor).any():
        print(f"Infinite values found in {name}")

def compute_metrics(pred: torch.Tensor, label: torch.Tensor, epsilon=1e-5):
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(1)
    if len(label.shape) == 3:
        label = label.unsqueeze(1)

    pred_ = pred.to(torch.bool)
    label_ = label.to(torch.bool)

    intersection = torch.logical_and(pred_, label_)
    union = torch.logical_or(pred_, label_)

    num_intersection = torch.sum(intersection, dim=(-2, -1))
    num_union = torch.sum(union, dim=(-2, -1))
    iou = safe_divide(num_intersection, num_union)
    dice = safe_divide(2 * num_intersection, torch.sum(pred_, dim=(-2, -1)) + torch.sum(label_, dim=(-2, -1)))

    iou = torch.mean(iou, dim=0)
    dice = torch.mean(dice, dim=0)
    iou = iou[-1].item() if iou.numel() > 1 else iou.item()
    dice = dice[-1].item() if dice.numel() > 1 else dice.item()

    e_tp = intersection
    e_fp = torch.logical_and(pred_, torch.logical_not(label_))
    e_fn = torch.logical_and(torch.logical_not(pred_), label_)
    e_tn = torch.logical_and(torch.logical_not(pred_), torch.logical_not(label_))

    n_tp = torch.count_nonzero(e_tp, dim=(0, -2, -1))
    n_fp = torch.count_nonzero(e_fp, dim=(0, -2, -1))
    n_fn = torch.count_nonzero(e_fn, dim=(0, -2, -1))
    n_tn = torch.count_nonzero(e_tn, dim=(0, -2, -1))

    precision = safe_divide(n_tp, (n_tp + n_fp))
    recall = safe_divide(n_tp, (n_tp + n_fn))
    specificity = safe_divide(n_tn, (n_tn + n_fp))
    f1 = safe_divide(2 * precision * recall, precision + recall)

    precision = precision[-1].item() if precision.numel() > 1 else precision.item()
    recall = recall[-1].item() if recall.numel() > 1 else recall.item()
    f1 = f1[-1].item() if f1.numel() > 1 else f1.item()
    specificity = specificity[-1].item() if specificity.numel() > 1 else specificity.item()

    return {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall,
            'f1': f1, 'specificity': specificity}

def compute_pixel_accuracy(pred, label):
    correct_pixels = (pred == label).sum().float()
    total_pixels = torch.tensor(label.numel(), dtype=torch.float, device=correct_pixels.device)
    accuracy = correct_pixels / total_pixels
    return accuracy.item()

### Testing Function ###
def test_single_volume(image, label, net, classes, patch_size=[128, 128], test_save_path=None, case=None, z_spacing=1):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y = slice_.shape
            if (x, y) != (patch_size[0], patch_size[1]):
                slice_ = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=3)
            input_tensor = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input_tensor)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

            if (x, y) != (patch_size[0], patch_size[1]):
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred
    else:
        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input_tensor), dim=1), dim=1).squeeze(0).cpu().detach().numpy()
        prediction = out

    metric_list = []
    results_folder = "./results_folder"
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, "all_cases_metrics.txt")

    # Class label assumed to be binary: GT=255 used as foreground
    for i in range(1, classes):
        metrics = compute_metrics(prediction == i, label == 255)
        metric_list.append(metrics)
        with open(results_file, 'a') as f:
            f.write(f"File: {case}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write("\n")

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))

    return metric_list


### Plotting Function ###
def plot_metrics(epoch, train_metrics, val_metrics, snapshot_path):
    # Create a separate folder for plots
    plots_path = os.path.join(snapshot_path, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    epochs = range(1, epoch+1)

    # We will plot the following metrics vertically in one figure:
    # 1) Accuracy
    # 2) Sensitivity(Recall)
    # 3) Specificity
    # 4) AUC
    # 5) Dice
    # 6) IoU
    # 7) Loss
    # 8) Learning Rate

    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(8, 24))
    fig.subplots_adjust(hspace=0.5)

    # Accuracy
    axes[0].plot(epochs, train_metrics['acc'], label='Train Acc', color='blue')
    axes[0].plot(epochs, val_metrics['acc'], label='Val Acc', color='orange')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Acc')
    axes[0].legend()
    axes[0].grid(True)

    # Sensitivity(Recall)
    axes[1].plot(epochs, train_metrics['recall'], label='Train Sensitivity', color='blue')
    axes[1].plot(epochs, val_metrics['recall'], label='Val Sensitivity', color='orange')
    axes[1].set_title('Sensitivity(Recall)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Sensitivity')
    axes[1].legend()
    axes[1].grid(True)

    # Specificity
    axes[2].plot(epochs, train_metrics['specificity'], label='Train Specificity', color='blue')
    axes[2].plot(epochs, val_metrics['specificity'], label='Val Specificity', color='orange')
    axes[2].set_title('Specificity')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Specificity')
    axes[2].legend()
    axes[2].grid(True)

    # AUC
    axes[3].plot(epochs, train_metrics['auc'], label='Train AUC', color='blue')
    axes[3].plot(epochs, val_metrics['auc'], label='Val AUC', color='orange')
    axes[3].set_title('AUC')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('AUC')
    axes[3].legend()
    axes[3].grid(True)

    # Dice
    axes[4].plot(epochs, train_metrics['dice'], label='Train Dice', color='blue')
    axes[4].plot(epochs, val_metrics['dice'], label='Val Dice', color='orange')
    axes[4].set_title('Dice')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Dice')
    axes[4].legend()
    axes[4].grid(True)

    # IoU
    axes[5].plot(epochs, train_metrics['iou'], label='Train IoU', color='blue')
    axes[5].plot(epochs, val_metrics['iou'], label='Val IoU', color='orange')
    axes[5].set_title('IoU')
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('IoU')
    axes[5].legend()
    axes[5].grid(True)

    # Loss
    axes[6].plot(epochs, train_metrics['loss'], label='Train Loss', color='blue')
    axes[6].plot(epochs, val_metrics['loss'], label='Val Loss', color='orange')
    axes[6].set_title('Loss')
    axes[6].set_xlabel('Epoch')
    axes[6].set_ylabel('Loss')
    axes[6].legend()
    axes[6].grid(True)

    # Learning Rate (only train)
    axes[7].plot(epochs, train_metrics['lr'], label='Learning Rate', color='green')
    axes[7].set_title('Learning Rate')
    axes[7].set_xlabel('Epoch')
    axes[7].set_ylabel('LR')
    axes[7].legend()
    axes[7].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_path, f'metrics_epoch{epoch}.png'))
    plt.close(fig)
