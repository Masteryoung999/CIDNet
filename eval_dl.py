import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I

def calculate_raw_metrics(pred, gt, data_range=None):
    """
    计算4通道RAW图像的PSNR和SSIM指标

    参数:
        pred (numpy.ndarray): 预测图像数组[H,W,4]
        gt (numpy.ndarray): 真实图像数组[H,W,4]
        data_range (float, optional): 像素值范围，默认自动判断

    返回:
        tuple: (PSNR值, SSIM值)

    异常:
        ValueError: 输入形状不匹配或非四通道
    """
    # 形状一致性验证
    if pred.shape != gt.shape:
        raise ValueError(f"形状不匹配: pred {pred.shape} vs gt {gt.shape}")

    # 通道数验证
    if pred.ndim != 3 or pred.shape[2] != 4:
        raise ValueError("输入必须是四通道图像 (HxWx4)")

    # 数据范围自动判断
    if data_range is None:
        if np.issubdtype(pred.dtype, np.integer):
            data_range = np.iinfo(pred.dtype).max
        else:
            data_range = 1.0  # 归一化数据假设

    # 转换为浮点型避免计算溢出
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    # 计算PSNR（自动处理多通道）
    psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)

    # 计算SSIM（指定通道轴）
    ssim = structural_similarity(gt, pred,
                                 win_size=7,
                                 channel_axis=-1,  # 最后一个维度是通道
                                 data_range=data_range)

    return float(psnr), float(ssim)

gt_root = '/data2/yyh/NTIRE_RAWSR2024/val_gt/'
output_root = '/data3/yyh/distillation_v2/results/mffssr_small'

psnr_list, ssim_list = [], []
for ind in tqdm(range(40)):
    gt = np.load(os.path.join(gt_root, f'{ind + 1}.npz'))
    raw_img_gt = gt["raw"].astype(np.float32)
    raw_max_gt = gt["max_val"]
    raw_img_gt = raw_img_gt / raw_max_gt

    output = np.load(os.path.join(output_root, f'{ind + 1}.npz'))
    raw_img_output = output["raw"].astype(np.float32)
    raw_max_output = output["max_val"]
    raw_img_output = raw_img_output / raw_max_output
    raw_img_output = np.clip(raw_img_output, 0, 1)


    psnr, ssim = calculate_raw_metrics(raw_img_output, raw_img_gt, data_range=1)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
print(psnr_list)
print(np.mean(psnr_list), np.mean(ssim_list))