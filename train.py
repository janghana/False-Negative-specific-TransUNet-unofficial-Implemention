import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_mrt1
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../datasets/brain_dataset_npz/train', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='MRT1', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_MRT1', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect')
parser.add_argument('--vit_name', type=str,
                    default='R50+ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--train_root_path', type=str, default='../../datasets/duct_part_npz/train', help='training set root path')
parser.add_argument('--val_root_path', type=str, default='../../datasets/duct_part_npz/val', help='validation set root path')
parser.add_argument('--train_split', type=str, default='duct_part_train_npz', help='train split txt file prefix')
parser.add_argument('--val_split', type=str, default='duct_part_val_npz', help='val split txt file prefix')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
args = parser.parse_args()

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    start_time = time.time()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'MRT1': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    trainer = {dataset_name: trainer_mrt1}
    trainer[dataset_name](args, net, snapshot_path)

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

# nohup python train.py --dataset MRT1 --train_root_path ../../datasets/duct_part_npz/train --val_root_path ../../datasets/duct_part_npz/val --list_dir ./lists/lists_MRT1 --train_split duct_part_train_npz --val_split duct_part_val_npz --num_classes 2 --max_epochs 150 --img_size 256 --batch_size 2 --vit_name R50+ViT-B_16 --n_gpu 2 --gpu_id "0,1" --base_lr 5e-4 > train.log 2>&1 &

# ls *.npz | sed 's/\.npz$//' > train.txt
