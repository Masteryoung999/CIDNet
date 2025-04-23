import os

import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.loss_functions import *
from utility.feature_transformation import spatial_similarity
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
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=1e-3')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay. default = 0')
    parser.add_argument('--loss', type=str, default='l2', choices=['l1','l2','smooth_l1','ssim','l2_ssim','nll', 'invnet'],
                        help='loss')
    parser.add_argument('--epoch', type=int, default=100, help='training epoches')
    parser.add_argument('--iterations', type=int, default=1, help='training iterations')
    parser.add_argument('--init', type=str, default='kn',choices=['kn', 'ku', 'xn', 'xu', 'edsr'],
                        help='which init scheme to choose.')
    parser.add_argument('--scheduler', type=str, default='cosine',choices=['cosine', 'reduce'],
                        help='which init scheduler to choose.')
    parser.add_argument('--init_scale', type=float, default=0.5)
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--clear', action='store_true', help='remove best psnr?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=2,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use. default=2018')
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
    parser.add_argument('--clip', type=float, default=0.1, help='gradient clip threshold')
    parser.add_argument('--reg_l1', type=float, default=0.)
    parser.add_argument('--reg_l2', type=float, default=0.)
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    opt = parser.parse_args()
    return opt

def init_params(net, init_type='kn', scale=0.1):
    print('use init scheme: %s' % init_type)
    if not init_type:
        return

    if init_type == 'net':
        net.init_params(init_type, scale)
    else:
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                if m._get_name() in ['BasicConv3dZeros', 'BasicConv2dZeros']:
                    init.constant_(m.weight, 0)
                elif init_type == 'kn':
                    init.kaiming_normal_(m.weight, mode='fan_in')
                elif init_type == 'ku':
                    init.kaiming_uniform_(m.weight, mode='fan_in')
                elif init_type == 'xn':
                    init.xavier_normal_(m.weight)
                elif init_type == 'xu':
                    init.xavier_uniform_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init.normal_(m.weight, std=1e-3)
            #     m.weight.data *= scale
            #     if m.bias is not None:
            #         init.constant_(m.bias, 0)

def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def init_criterion(loss):
    if loss == 'l2':
        criterion = nn.MSELoss()
    elif loss == 'l1':
        criterion = nn.L1Loss()
    elif loss == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif loss == 'rbsformer':
        criterion = RBSFormerLoss()
    elif loss == 'rbsformer_w1':
        criterion = RBSFormerLoss(lambda_freq=1)
    elif loss == 'psnr_rbs':
        criterion = PSNRRBS()
    else:
        criterion = nn.MSELoss()
    return criterion

def get_summary_writer(log_dir, prefix=None):
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    return writer

class Engine(object):
    def __init__(self, opt, engine_teacher = None):
        self.engine_teacher = engine_teacher
        self.opt = opt
        self.prefix = opt.prefix
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None
        self.clip_previous_s = opt.clip
        self.clip_previous_t = opt.clip
        self.alpha = 0.5
        self.beta = 1 - self.alpha
        self.features = [1,2,3]  # features selected,指定的特征层做特征损失
        self.__setup()

    def __setup(self):
        self.basedir = os.path.join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        cuda = not self.opt.no_cuda
        self.device = 'cuda' if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch.lower()]()

        """Params Init"""
        init_params(self.net, init_type=self.opt.init, scale=self.opt.init_scale)


        if cuda and self.opt.world_size > 1:
            torch.cuda.set_device(self.opt.rank)
            self.net.to(self.device)
            self.net = DDP(self.net, device_ids=[self.opt.rank])
        else:
            self.net.to(self.device)

        """Loss Function"""
        self.criterion = init_criterion(self.opt.loss)
        if cuda:
            self.criterion = self.criterion.to(self.device)
        print('criterion: ', self.criterion)


        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            if self.opt.rank == 0:
                self.writer = get_summary_writer(os.path.join(self.basedir, 'logs', self.opt.prefix))

        """Optimization Setup"""
        if self.opt.world_size > 1:
            self.optimizer = torch.optim.AdamW(
                self.net.module.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)
        else:
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)
        if self.opt.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.opt.iterations, eta_min=1e-5)
        elif self.opt.scheduler == 'reduce':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
            )
        else:
            raise ValueError

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            print('==> Loading checkpoint...')
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
            # print(self.net)

    def load(self, resumePath=None, load_opt=True):
        model_best_path = os.path.join(self.basedir, self.prefix, 'model_latest.pth')
        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path, map_location=self.device)
        #### comment when using memnet
        self.epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        self.iteration = checkpoint['iteration'] if 'iteration' in checkpoint.keys() else -1
        self.best_psnr = checkpoint['best_psnr'] if 'best_psnr' in checkpoint.keys() else 100


        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optimizer.param_groups[0]['capturable'] = True
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if 'net' in checkpoint.keys():
            model_dict = checkpoint['net']
        else:
            model_dict = checkpoint['state_dict']
            pretrained_dict = {k.replace('net.', ''): v for k, v in model_dict.items()}
            model_dict = pretrained_dict
        self.get_net().load_state_dict(model_dict)
        print(f"==> Resuming from {self.epoch} epoch, best PSNR {self.best_psnr}")

        if self.opt.clear:
            print('===> Clearing')
            self.epoch, self.iteration, self.best_psnr = 0, 0, 0

    def get_net(self):
        if self.opt.world_size > 1 and not self.opt.no_cuda:
            return self.net.module
        else:
            return self.net


    """Forward Functions"""
    def forward_chop(self, x, base=32):
        if len(x.shape) == 5:
            n, c, b, h, w = x.size()
        else:
            n, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2

        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)

        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)

        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1

        output /= output_w

        return output

    def forward_slice(self, inputs, base=31, step=31):
        if len(inputs.shape) == 5:
            n, c, b, h, w = inputs.size()
        else:
            n, b, h, w = inputs.size()
        output = torch.zeros_like(inputs)
        cnt = torch.zeros_like(inputs)
        for bb in range(0, b-base+1, step):
            x = inputs[...,bb:bb+base,:,:]
            output_current = self.net(x)
            if isinstance(output_current, tuple):
                output_current = output_current[0]
            output[...,bb:bb+base,:,:] += output_current
            cnt[...,bb:bb+base,:,:] += 1
        if b % step != 0:
            x = inputs[..., b-base:b, :, :]
            output_current = self.net(x)
            if isinstance(output_current, tuple):
                output_current = output_current[0]
            output[...,b-base:b,:,:] += output_current
            cnt[...,b-base:b,:,:] += 1
        output = output / cnt
        return output

    def forward_chop_slice(self, inputs, base=31, win_size=32):
        if len(inputs.shape) == 5:
            n, c, b, h, w = inputs.size()
            output = torch.zeros_like(inputs)
            cnt = torch.zeros_like(inputs)
            for bb in range(0, b - base + 1, base):
                x = inputs[:, :, bb:bb + base, :, :]
                if h % win_size != 0 or w % win_size != 0:
                    output[:, :, bb:bb + base, :, :] += self.forward_chop(x, win_size)
                else:
                    output[:, :, bb:bb + base, :, :] += self.net(x)
                cnt[:, :, bb:bb + base, :, :] += 1
            if b % base != 0:
                x = inputs[:, :, b - base:b, :, :]
                if h % win_size != 0 or w % win_size != 0:
                    output[:, :, b - base:b, :, :] += self.forward_chop(x)
                else:
                    output[:, :, b - base:b, :, :] += self.net(x)
                cnt[:, :, b - base:b, :, :] += 1
            output = output / cnt
        else:
            n, b, h, w = inputs.size()
            output = torch.zeros_like(inputs)
            cnt = torch.zeros_like(inputs)
            # pt = torch.load('output.pt')
            # output, cnt = pt['output'], pt['cnt']
            for bb in range(0, b - base + 1, base):
                # torch.save({'output':output.detach().cpu(), 'cnt':cnt.detach().cpu()}, 'output.pt')
                # print(bb)
                x = inputs[:, bb:bb + base, :, :]
                if h % win_size != 0 or w % win_size != 0:
                    output[:, bb:bb + base, :, :] += self.forward_chop(x, win_size)
                else:
                    output[:, bb:bb + base, :, :] += self.net(x)
                cnt[:, bb:bb + base, :, :] += 1
            if b % base != 0:
                x = inputs[..., b - base:b, :, :]
                if h % win_size != 0 or w % win_size != 0:
                    output[..., b - base:b, :, :] += self.forward_chop(x)
                else:
                    output[..., b - base:b, :, :] += self.net(x)
                cnt[..., b - base:b, :, :] += 1
            print('ok')
            output = output / cnt
        output[output < 0] = 1e-6
        return output

    def forward(self, inputs):
        if self.opt.chop and not self.opt.slice:
            outputs = self.forward_chop(inputs)
        elif not self.opt.chop and self.opt.slice:
            outputs = self.forward_slice(inputs)
        elif self.opt.chop and self.opt.slice:
            outputs = self.forward_chop_slice(inputs)
        else:
            if self.net.training:
                outputs, featlist = self.net(inputs, return_feats=True)
                return outputs, featlist
            else:
                outputs = self.get_net()(inputs)
                return outputs

    def __step(self, train, inputs, targets):
        loss_info = None
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = 0
        time_start = time.time()
        aggregated_student_fms = []
        aggregated_teacher_fms = []

        if train:

            outputs_s, featlist_s = self.forward(inputs)
            with torch.no_grad():
                # self.engine_teacher.net.eval()
                outputs_t, featlist_t = self.engine_teacher.forward(inputs)
            #     outputs_s = outputs_t
            # outputs_t = outputs_s
            loss_ds = self.criterion(outputs_s, targets)
            loss_ts = self.criterion(outputs_s, outputs_t)

            # print(len(featlist_s))
            # print(len(featlist_t))
            # 源代码self.features = [1,2,3]
            student_fms = [featlist_s[ind] for ind in [0,1,2,3]]  # 从学生网络的特征图中选择指定的特征层
            teacher_fms = [featlist_t[ind] for ind in [0,1,2,3]]  # 从教师网络的特征图中选择指定的特征层

            # 特征损失
            assert len(student_fms) == len(teacher_fms)
            aggregated_student_fms = ([spatial_similarity(fm) for fm in student_fms])
            aggregated_teacher_fms = ([spatial_similarity(fm) for fm in teacher_fms])

            # 原文是l1loss
            l1_loss = nn.L1Loss()
            loss_feature = sum(l1_loss(fm1, fm2) for fm1, fm2 in zip(aggregated_student_fms, aggregated_teacher_fms))
            # 如果用RBSFormerLoss
            # loss_feature = sum(self.criterion(fm1, fm2) for fm1, fm2 in zip(aggregated_student_fms, aggregated_teacher_fms))

            # loss = self.alpha * loss_ds + self.beta * loss_ts
            loss = self.alpha * loss_ds + self.beta * loss_ts + loss_feature
        else:
            outputs_s = self.forward(inputs)
            loss = self.criterion(outputs_s, targets)

        if isinstance(loss, tuple):
            loss_info = [t.item() for t in loss]
            loss = sum(loss)


        loss = loss + l2_regularization(self.net, self.opt.reg_l2) + l1_regularization(self.net, self.opt.reg_l1)
        if isinstance(outputs_s, (tuple, list)):
            outputs_s = outputs_s[0]


        time_end = time.time()

        if train:
            loss.backward()
        loss_data += loss.item()

        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()
            self.scheduler.step()

        timecost = time_end - time_start
        return outputs_s, loss_data, total_norm, timecost, loss_info

    """Training Functions"""
    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.net.train()
        avg_loss, avg_loss1, avg_loss2 = 0, 0, 0
        train_loss, train_loss1, train_loss2 = 0, 0, 0
        for batch_idx, (inputs, targets, meta_info) in enumerate(train_loader):
            if not self.opt.no_cuda:
                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.cuda(self.opt.rank, non_blocking=True)
                targets = targets.cuda(self.opt.rank, non_blocking=True)
            outputs, loss_data, total_norm, time_cost, loss_info = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx + 1)

            if not self.opt.no_log and self.opt.rank == 0:
                self.writer.add_scalar(
                    os.path.join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    os.path.join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)
                self.writer.add_scalar(
                    os.path.join(self.prefix, 'total_norm'), total_norm, self.iteration)
                if loss_info is not None:
                    train_loss1 = train_loss1 + loss_info[0]
                    train_loss2 = train_loss2 + loss_info[1]
                    avg_loss1 = train_loss1 / (batch_idx + 1)
                    avg_loss2 = train_loss2 / (batch_idx + 1)
                    self.writer.add_scalar(
                        os.path.join(self.prefix, 'avg_loss1'), avg_loss1, self.iteration)
                    self.writer.add_scalar(
                        os.path.join(self.prefix, 'avg_loss2'), avg_loss2, self.iteration)
            self.iteration += 1
            if loss_info is None:
                progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e'% (avg_loss, loss_data, total_norm))
            else:
                progress_bar(batch_idx, len(train_loader), 'AL:%.2e|AL1:%.2e|AL2:%.2e|L1:%.2e|L2:%.2e|Norm:%.2e'
                             % (avg_loss, avg_loss1, avg_loss2, loss_info[0], loss_info[1], total_norm))


        self.epoch += 1
        if not self.opt.no_log and self.opt.rank == 0:
            self.writer.add_scalar(
                os.path.join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

    def limit_output(self, outputs):
        outputs = torch.clip(outputs, 0, 1)
        # mask = torch.where(torch.isnan(outputs), torch.full_like(outputs, 0), torch.full_like(outputs, 1))
        # outputs[mask == 0] = torch.mean(outputs[mask == 1])
        # outputs = torch.minimum(outputs, torch.Tensor([1]).to(outputs.device))
        # outputs = torch.maximum(outputs, torch.Tensor([0.00001]).to(outputs.device))
        return outputs

    """Validation Functions"""
    def validate(self, valid_loader, name):
        device = self.device
        self.net.to(device)
        self.criterion = self.criterion.to(device)
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        print('\n[i] Eval dataset {}...'.format(name))
        with torch.no_grad():
            for batch_idx, (inputs, targets, meta_info) in enumerate(valid_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, loss_data, _, time_cost, loss_info = self.__step(False, inputs, targets)
                outputs = self.limit_output(outputs)
                psnr = np.mean(cal_bwpsnr(outputs, targets))
                validate_loss += loss_data
                avg_loss = validate_loss / (batch_idx + 1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx + 1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                             % (avg_loss, avg_psnr))

        if not self.opt.no_log and self.opt.rank == 0:
            self.writer.add_scalar(
                os.path.join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                os.path.join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    """Model Saving Functions"""
    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = os.path.join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
            'best_psnr': self.best_psnr,
        }

        state.update(kwargs)

        if not os.path.isdir(os.path.join(self.basedir, self.prefix)):
            os.makedirs(os.path.join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

        # if self.opt.world_size <= 1:
        if self.opt.rank == 0:
            self.writer.file_writer.flush()
            from tensorboard.backend.event_processing import event_accumulator
            p = self.writer.file_writer.get_logdir()
            f = [t for t in os.listdir(p) if t.startswith('events')]
            ea = event_accumulator.EventAccumulator(os.path.join(p, f[0]))  # 初始化EventAccumulator对象
            ea.Reload()
            for k in ea.scalars.Keys():
                data_log = ea.scalars.Items(k)
                data_log = [t.value for t in data_log]
                plt.figure()
                plt.plot(data_log)
                plt.xlabel(k)
                plt.savefig(os.path.join(p, k.split('/')[-1]+'.png'))
                plt.close()
