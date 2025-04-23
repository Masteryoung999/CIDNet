import random
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
import time
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms.functional as TF

from scipy.ndimage import uniform_filter1d


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def minmax_normalize(array):
    array = array.astype(np.float32)
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def torch2numpy(hsi, use_2dconv):
    if use_2dconv:
        R_hsi = hsi.data[0].cpu().permute((1, 2, 0)).contiguous().numpy()
    else:
        R_hsi = hsi.data[0].cpu()[0, ...].permute((1, 2, 0)).contiguous().numpy()
    return R_hsi


""" Visualize """

def Visualize3D(data, frame = 0):
    data = np.squeeze(data)
    data[frame, ...] = minmax_normalize(data[frame, ...])
    plt.imshow(data[frame, :, :], cmap='gray')  # shows 256x256 image, i.e. 0th frame
    plt.show()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

term_width = 30
# TOTAL_BAR_LENGTH = 25.
TOTAL_BAR_LENGTH = 1.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')


    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    L.append(' | Remain: %.3f h' % (step_time*(total-current-1) / 3600))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    sys.stdout.write(' %d/%d ' % (current+1, total))
    sys.stdout.flush()


""" learning rate """
def adjust_learning_rate(optimizer, lr):
    print('Adjust Learning Rate => %.4e' %lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def display_learning_rate(optimizer):
    lrs = []
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        print('learning rate of group %d: %.4e' % (i, lr))
        lrs.append(lr)
    return lrs




""" Augmentation """
def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :]

    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    # we apply spectrum reversal for training 3D CNN, e.g. QRNN3D.
    # disable it when training 2D CNN, e.g. MemNet
    # if random.random() < 0.5:
    #     image = image[::-1, :, :]

    return np.ascontiguousarray(image)

""" Crop """


def crop_cxhxw_images(
        images: List[np.ndarray],
        crop_height: int,
        crop_width: int,
        x: int = None,
        y: int = None
) -> List[np.ndarray]:
    """
    对CxHxW格式的numpy数组图像进行相同位置裁剪

    参数：
    images: 包含numpy数组的图像列表，形状为 (Channels, Height, Width)
    crop_height: 裁剪区域高度 (必须 ≤ 原图高度)
    crop_width: 裁剪区域宽度 (必须 ≤ 原图宽度)
    x: 水平起始坐标 (None表示自动计算水平中心)
    y: 垂直起始坐标 (None表示自动计算垂直中心)

    返回：
    裁剪后的图像列表，形状为 (Channels, crop_height, crop_width)
    """
    # 参数校验
    if not isinstance(images, (list, tuple)) or len(images) == 0:
        raise ValueError("输入必须是包含至少一个numpy数组的列表")

    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("裁剪尺寸必须为正整数")

    # 统一校验所有图像
    base_channels, base_h, base_w = None, None, None
    for idx, img in enumerate(images):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"第 {idx} 个元素不是numpy数组")
        if img.ndim != 3:
            raise ValueError(f"第 {idx} 个数组维度错误，应为3维 (CxHxW)")
        if base_h is None:
            base_channels, base_h, base_w = img.shape
        else:
            if img.shape[1:] != (base_h, base_w):
                raise ValueError(f"第 {idx} 个图像尺寸 {img.shape} 与其他图像不一致")


    # 边界校验
    if x < 0 or y < 0:
        raise ValueError(f"裁剪起始坐标不能为负数 (x={x}, y={y})")
    if (x + crop_width) > base_w:
        raise ValueError(f"水平方向越界: 原图宽度 {base_w} < 裁剪终止位置 {x + crop_width}")
    if (y + crop_height) > base_h:
        raise ValueError(f"垂直方向越界: 原图高度 {base_h} < 裁剪终止位置 {y + crop_height}")

    # 执行裁剪 (所有图像共享相同坐标)
    return [img[:, y:y + crop_height, x:x + crop_width] for img in images]


def crop_center(img_list, cropx, cropy):
    _, y, x = img_list[0].size
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return crop_cxhxw_images(img_list, cropy, cropx, starty, startx)

import numpy as np
from PIL import Image

def crop_rand(img_list, crop_height, crop_width):
    # 将每个 PIL 图像转换为 numpy 数组，并调整维度为 (Channels, Height, Width)
    img_list = [np.array(img.convert('RGB')).transpose(2, 0, 1) for img in img_list]
    
    # 获取图像的宽和高
    width, height = img_list[0].shape[2], img_list[0].shape[1]  # shape 为 (Channels, Height, Width)
    
    # 计算随机裁剪位置，确保不会超出图像边界
    x = random.randint(0, max(0, width - crop_width))  # 确保 x 不越界
    y = random.randint(0, max(0, height - crop_height))  # 确保 y 不越界

    outputs = crop_cxhxw_images(img_list, crop_height, crop_width, x, y)
    # 调用裁剪函数
    return outputs


def identity(data):
    return data

def get_all_files(path):
    """
    获取指定路径下所有文件的绝对路径（包含子文件夹）

    参数：
    path (str/Path): 输入的目录路径

    返回：
    list: 包含所有文件绝对路径的列表

    示例：
    >>> get_all_files("my_folder")
    ['/absolute/path/my_folder/file1.txt',
     '/absolute/path/my_folder/subdir/file2.jpg']
    """
    path = Path(path).resolve()  # 转换为绝对路径

    if not path.exists():
        return []
    if not path.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {path}")

    return [str(file_path) for file_path in path.rglob("*") if file_path.is_file() and file_path.name.endswith(".png")]

def to_tensor_transform(img_list):
    outputs = [TF.to_tensor(img) for img in img_list]
    return outputs

