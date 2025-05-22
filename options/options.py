import os

import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utility.loss_functions import *
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
from scipy.io import savemat
import models
from utility import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = [int(str_arg) for str_arg in str_args if int(str_arg) >= 0]
        return parsed_args

    parser.add_argument('--prefix', '-p', type = str, default='temp', help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=False, choices=model_names,
                        help = 'model architecture: ' + ' | '.join(model_names), default='none' )
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=1e-3')
    parser.add_argument('--batchSize', type=int, default=8, help='batch size for training')
    parser.add_argument('--wd', type=float, default=0, help='weight decay. default = 0')
    parser.add_argument('--loss', type=str, default='CIDLoss', choices=['l1','l2','smooth_l1','ssim','l2_ssim','nll', 'invnet', 'CIDLoss'],
                        help='loss')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--scheduler', type=str, default='cosine',choices=['cosine', 'reduce'],
                        help='which init scheduler to choose.')
    parser.add_argument('--epoch', type=int, default=4000, help='training epoches')
    parser.add_argument('--iterations', type=int, default=1, help='training iterations')
    parser.add_argument('--init', type=str, default='kn',choices=['kn', 'ku', 'xn', 'xu', 'edsr'],
                        help='which init scheme to choose.')
    parser.add_argument('--init_scale', type=float, default=0.5)
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--clear', action='store_true', help='remove best psnr?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=8,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2025,
                        help='random seed to use. default=2025')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')
    parser.add_argument('--slice', action='store_true',
                            help='forward chop')
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--dataname', '-d', type=str,
                        default='', help='data root')
    parser.add_argument('--clip', type=float, default=1, help='gradient clip threshold')
    parser.add_argument('--gpu-ids', type=str, default='2', help='gpu ids, ex:0,1,2,3')
    parser.add_argument('--reg_l1', type=float, default=0.)
    parser.add_argument('--reg_l2', type=float, default=0.)

    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')


    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=bool, default=True, help='if gradient fluctuates too much, turn-on it')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=0)
    parser.add_argument('--D_weight',  type=float, default=0)
    parser.add_argument('--E_weight',  type=float, default=0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)

    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--lol_v1', type=bool, default=True)
    parser.add_argument('--lolv2_real', type=bool, default=False)
    parser.add_argument('--lolv2_syn', type=bool, default=False)
    parser.add_argument('--lol_blur', type=bool, default=False)
    parser.add_argument('--SID', type=bool, default=False)
    parser.add_argument('--SICE_mix', type=bool, default=False)
    parser.add_argument('--SICE_grad', type=bool, default=False)

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt