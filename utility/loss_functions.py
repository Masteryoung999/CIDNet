import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.vgg_arch import VGGFeatureExtractor, Registry
from utility.loss_utils import *
import torch.fft as fft


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight


class CIDLoss(nn.Module):
    def __init__(self, L1_weight, D_weight, E_weight, P_weight):
        super(CIDLoss, self).__init__()
        self.P_weight = P_weight
        self.L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
        self.D_loss = SSIM(weight=D_weight).cuda()
        self.E_loss = EdgeLoss(loss_weight=E_weight).cuda()
        self.P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = 1.0 ,criterion='mse').cuda()

    def forward(self, pred, target):
        loss = self.L1_loss(pred, target) + self.D_loss(pred, target) + self.E_loss(pred, target) + self.P_weight * self.P_loss(pred, target)[0]
        return loss

    
class RBSFormerLoss(nn.Module):
    """
    RBSFormer的复合损失函数
    包含Charbonnier损失和频率域L1损失
    """

    def __init__(self, eps=1e-3, lambda_freq=0.5):
        """
        Args:
            eps (float): Charbonnier损失的平滑系数
            lambda_freq (float): 频率损失的权重系数
        """
        super().__init__()
        self.eps = eps
        self.lambda_freq = lambda_freq

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 网络输出 [B, C, H, W]
            target (Tensor): 目标图像 [B, C, H, W]
        Returns:
            loss (Tensor): 综合损失值
        """
        # 空间域Charbonnier损失
        char_loss = self.charbonnier_loss(pred, target)

        # 频率域L1损失
        freq_loss = self.frequency_loss(pred, target)

        # 综合损失
        total_loss = char_loss + self.lambda_freq * freq_loss
        return total_loss

    def charbonnier_loss(self, pred, target):
        """ 鲁棒的L1损失，带epsilon平滑 """
        diff = pred - target
        return torch.sqrt(diff ** 2 + self.eps ** 2).mean()

    def frequency_loss(self, pred, target):
        """ 频域L1损失 (实部+虚部) """
        # 快速傅里叶变换
        pred = pred.contiguous().to(torch.float32)
        target = target.contiguous().to(torch.float32)

        pred_fft = fft.fft2(pred)
        target_fft = fft.fft2(target)

        # 分离实部虚部
        pred_real, pred_imag = pred_fft.real, pred_fft.imag
        target_real, target_imag = target_fft.real, target_fft.imag

        # 计算L1损失

        loss_real = F.l1_loss(pred_real, target_real)
        loss_imag = F.l1_loss(pred_imag, target_imag)

        return (loss_real + loss_imag) / 2  # 平均实虚损失


class comLoss(nn.Module):
    def __init__(self, E_weight, P_weight):
        super(CIDLoss, self).__init__()
        self.P_weight = P_weight
        self.RBSFormerLoss= RBSFormerLoss.cuda()
        self.E_loss = EdgeLoss(loss_weight=E_weight).cuda()
        self.P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = 1.0 ,criterion='mse').cuda()

    def forward(self, pred, target):
        loss = self.RBSFormerLoss(pred, target) + self.E_loss(pred, target) + self.P_weight * self.P_loss(pred, target)[0]
        return loss