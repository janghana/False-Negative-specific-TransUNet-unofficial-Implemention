# False-Negative-specific-TransUNet-unofficial-Implemention

This repository is an unofficial implementation of TransUNet with modifications to reduce false negatives in segmentation tasks, especially tailored for challenging structures like bile ducts. Building upon the original [TransUNet](https://arxiv.org/pdf/2102.04306.pdf) architecture, this version introduces a customized Dice-based loss function (inspired by Tversky loss) that allows adjusting sensitivity to False Negatives, thereby encouraging the model to be more aggressive in capturing small, difficult-to-segment anatomical structures.

## Key Modifications

1. **Dice-based Loss with FP/FN Weighting:**  
   We integrated a modified Dice loss function that includes parameters `alpha` and `beta` to control the relative penalties for False Positives (FP) and False Negatives (FN). By tuning these parameters, the model can strike a better balance between missing subtle targets (FN) and erroneously predicting structures where none exist (FP).

2. **Binary Label Preprocessing:**  
   To ensure consistent training and validation, labels are normalized to binary classes (0 or 1) so that the model focuses on the presence or absence of the target structure. This helps prevent indexing errors and aligns the loss function with the intended segmentation objective.

3. **Flexible Training Configuration:**  
   Paths to training and validation data, number of GPUs, batch size, and other parameters can now be passed via command-line arguments (argparse), reducing the need for hardcoded paths and splits.

## Usage

### 1. Download Pre-trained ViT Models
We rely on pre-trained ViT weights from Google:
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
mkdir -p ../model/vit_checkpoint/imagenet21k
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare Data
Follow the instructions in ./datasets/README.md to prepare your dataset. Ensure that your labels are binarized if not already. The code will also handle binarization during dataset loading.

### 3. Environment Setup

Create a Python 3.7 environment and install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Training
Use the following command to train the model. Specify GPUs, dataset paths, and other parameters via command line:

```bash
nohup python train.py \
    --dataset MRT1 \
    --train_root_path ../../datasets/duct_part_npz/train \
    --val_root_path ../../datasets/duct_part_npz/val \
    --list_dir ./lists/lists_MRT1 \
    --train_split train \
    --val_split val \
    --num_classes 2 \
    --max_epochs 150 \
    --img_size 256 \
    --batch_size 2 \
    --vit_name R50+ViT-B_16 \
    --n_gpu 2 \
    --gpu_id "0,1" \
    --base_lr 5e-4 > train.log 2>&1 &
```


This command:
- Uses GPUs 0 and 1,
- Trains for 150 epochs or more,
- Uses a base learning rate of 5e-4,
- Adjusts the Tversky-like loss parameters internally to handle false negatives.


### 5. TensorBoard and Plotting
While training, metrics will be logged to TensorBoard and plots will be periodically saved in the snapshot directory. For TensorBoard:

```bash
tensorboard --logdir=../model/TU_MRT1256/TU_pretrain_R50+ViT-B_16_skip3_.../log --port=6006
```
Access http://localhost:6006 in your browser.

The PNG plots for accuracy, sensitivity, specificity, AUC, and loss are saved under the same snapshot_path directory where the model is stored.

### 6. Testing
After training, you can run the test script (adapt your paths and model accordingly):

```bash
python test.py --dataset MRT1 --vit_name R50+ViT-B_16
```


## References

- https://github.com/Beckschen/TransUNet
- https://github.com/google-research/vision_transformer
- https://github.com/jeonsworld/ViT-pytorch
- https://github.com/qubvel-org/segmentation_models.pytorch

Citation

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L. and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
