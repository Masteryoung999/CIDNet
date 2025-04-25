import os

import matplotlib.pyplot as plt
import lpips as lpips_fn
import torch
import torch.nn.init as init
from torch import nn

from utility.loss_functions import *
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
from scipy.io import savemat
import models
from measure import metrics
from utility import *


def init_params(net, init_type='kn', scale=0.1):
    print('use init scheme: %s' % init_type)
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
    elif loss == 'CIDLoss':
        criterion = CIDLoss(L1_weight=0, D_weight=0, E_weight=0, P_weight=0.01)
    else:
        criterion = nn.MSELoss()
    return criterion

def get_summary_writer(log_dir, prefix=None):
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    return writer

class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None
        self.clip_previous = opt.clip
        self.L1_loss = None
        self.E_loss = None
        self.D_loss = None
        self.P_loss = None
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

        if cuda and len(self.opt.gpu_ids) > 1:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            self.net.to(self.device)
            self.net = nn.DataParallel(self.net, device_ids=self.opt.gpu_ids)
        if cuda and len(self.opt.gpu_ids) == 1:
            # print(torch.cuda.device_count())  # 看你机器有几张卡
            # print(self.opt.gpu_ids)           # 看你传进来的 ID 是什么
            torch.cuda.set_device(self.opt.gpu_ids[0])
            self.net.to(self.device)
        """Loss Function"""
        self.criterion = init_criterion(self.opt.loss)
        # self.L1_loss, self.D_loss, self.E_loss, self.P_loss = init_criterion(
        #     self.opt.L1_weight, self.opt.D_weight, self.opt.E_weight, self.opt.P_weight)
        if cuda:
            self.criterion = self.criterion.to(self.device)
        print('criterion: ', self.criterion)


        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs', self.opt.prefix))

        """Optimization Setup"""
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.opt.iterations, eta_min=1e-7)

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
        ####
        # model_dict = self.get_net().state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict and k.find('shift_attn_mask') == -1}
        # model_dict.update(pretrained_dict)
        if 'net' in checkpoint.keys():
            model_dict = checkpoint['net']
        else:
            model_dict = checkpoint['state_dict']
            pretrained_dict = {k.replace('net.', ''): v for k, v in model_dict.items()}
            model_dict = pretrained_dict
        self.get_net().load_state_dict(model_dict)
        # self.get_net().load_state_dict(checkpoint['net'], strict=False)
        print(f"==> Resuming from {self.epoch} epoch. Best PSNR: {self.best_psnr}")

        if self.opt.clear:
            print('===> Clearing')
            self.epoch, self.iteration, self.best_psnr = 0, 0, 0


    def get_net(self):
        if len(self.opt.gpu_ids) > 1 and not self.opt.no_cuda:
            return self.net.module
        else:
            return self.net


    """Forward Functions"""
    def forward(self, inputs):
        if self.net.training:
            outputs = self.net(inputs)
        else:
            outputs = self.get_net()(inputs)
        return outputs

    def __step(self, train, inputs, targets):
        loss_info = None
        if self.opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01, norm_type=2)

        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = 0
        time_start = time.time()
        if self.opt.gamma:
            gamma = random.randint(self.opt.start_gamma, self.opt.end_gamma) / 100.0
            output_rgb = self.forward(inputs ** gamma)  
        else:
            output_rgb = self.forward(inputs)  
        if isinstance(output_rgb, (tuple, list)):
                    output_rgb = output_rgb[0]
        gt_rgb = targets
        output_hvi = self.net.HVIT(output_rgb)
        gt_hvi = self.net.HVIT(gt_rgb)

        loss_hvi = self.criterion(output_hvi, gt_hvi)
        loss_rgb = self.criterion(output_rgb, gt_rgb)
        loss = loss_rgb + self.opt.HVI_weight * loss_hvi
        # loss = self.criterion(output_rgb, gt_rgb)

        if isinstance(loss, tuple):
            loss_info = [t.item() for t in loss]
            loss = sum(loss)
        # loss = loss + l2_regularization(self.net, self.opt.reg_l2) + l1_regularization(self.net, self.opt.reg_l1)
        
        
        time_end = time.time()

        if train:
            loss.backward()
        loss_data += loss.item()

        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()
            self.scheduler.step()

        timecost = time_end - time_start
        return output_rgb, loss_data, total_norm, timecost, loss_info

    """Training Functions"""
    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.net.train()
        avg_loss, avg_loss1, avg_loss2 = 0, 0, 0
        train_loss, train_loss1, train_loss2 = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs, loss_data, total_norm, time_cost, loss_info = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx + 1)

            if not self.opt.no_log:
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
        if not self.opt.no_log:
            self.writer.add_scalar(
                os.path.join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)
            
        # now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        # with open(f"./results/training/metrics{now}.md", "w") as f:
        #     f.write("dataset: "+ f"{self.opt.dataname}" + "\n")  
        #     f.write(f"lr: {self.opt.lr}\n")  
        #     f.write(f"HVI_weight: {self.opt.HVI_weight}\n")  
        #     f.write(f"L1_weight: {self.opt.L1_weight}\n")  
        #     f.write(f"D_weight: {self.opt.D_weight}\n")  
        #     f.write(f"E_weight: {self.opt.E_weight}\n")  
        #     f.write(f"P_weight: {self.opt.P_weight}\n")  
        #     f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        #     f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        #     for i in range(len(psnr)):
        #         f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | { psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")  
        

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
            # n = len(valid_loader)
            loss_fn = lpips_fn.LPIPS(net='alex')
            psnr = []
            ssim = []
            lpips = []

            for batch_idx, (inputs, targets) in enumerate(valid_loader):

                inputs, targets = inputs.to(device), targets.to(device)
                outputs, loss_data, _, time_cost, loss_info = self.__step(False, inputs, targets)
                outputs = self.limit_output(outputs)

                # LOL three subsets
                if self.opt.lol_v1:
                    output_folder = 'LOLv1/'
                    label_dir = self.opt.data_valgt_lol_v1
                # if self.opt.lolv2_real:
                #     output_folder = 'LOLv2_real/'
                #     label_dir = self.opt.data_valgt_lolv2_real
                # if self.opt.lolv2_syn:
                #     output_folder = 'LOLv2_syn/'
                #     label_dir = self.opt.data_valgt_lolv2_syn
                
                # # LOL-blur dataset with low_blur and high_sharp_scaled
                # if self.opt.lol_blur:
                #     output_folder = 'LOL_blur/'
                #     label_dir = self.opt.data_valgt_lol_blur
                    
                # if self.opt.SID:
                #     output_folder = 'SID/'
                #     label_dir = self.opt.data_valgt_SID
                #     npy = True
                # if self.opt.SICE_mix:
                #     output_folder = 'SICE_mix/'
                #     label_dir = self.opt.data_valgt_SICE_mix
                #     norm_size = False
                # if self.opt.SICE_grad:
                #     output_folder = 'SICE_grad/'
                #     label_dir = self.opt.data_valgt_SICE_grad
                #     norm_size = False

                im_dir = self.opt.val_folder + output_folder + '*.png'

                # avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=True)
                
                score_psnr, score_ssim, score_lpips = metrics(outputs, targets, use_GT_mean=True, loss_fn = loss_fn)
                psnr.append(score_psnr)
                ssim.append(score_ssim)
                lpips.append(score_lpips.item())

                avg_psnr = sum(psnr)/len(psnr)
                avg_ssim = sum(ssim)/len(ssim)
                avg_lpips = sum(lpips)/len(lpips)
            
                
                
                # psnr = np.mean(cal_bwpsnr(outputs, targets))
                validate_loss += loss_data
                avg_loss = validate_loss / (batch_idx + 1)

                # total_psnr += psnr
                # avg_psnr = total_psnr / (batch_idx + 1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | SSIM: %.4f | LPIPS: %.4f'
                             % (avg_loss, avg_psnr, avg_ssim, avg_lpips))
            print("")
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))

        if not self.opt.no_log:
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