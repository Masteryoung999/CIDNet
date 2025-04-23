import numpy as np
from .blur import apply_psf, add_blur
from .noise import add_natural_noise, add_gnoise, add_heteroscedastic_gnoise
from .imutils import downsample_raw, convert_to_tensor
import random
import os
from contextlib import contextmanager

@contextmanager
def fixed_seed(seed):
    # 保存原始状态
    orig_random_state = random.getstate()
    orig_np_state = np.random.get_state()

    # 设置固定种子
    random.seed(seed)
    np.random.seed(seed)

    try:
        yield  # 执行代码块
    finally:
        # 恢复原始状态
        random.setstate(orig_random_state)
        np.random.set_state(orig_np_state)

def data_augment(images_list):
    """
    对图像列表进行一致性的随机旋转/翻转增强
    输入：
        images_list: List[np.array] - 包含多个CxHxW格式图像的列表
    输出：
        List[np.array] - 应用相同增强后的图像列表
    """
    # 随机选择增强方式
    choice = random.choice([
        'rot90_1',  # 旋转90度
        'rot90_2',  # 旋转180度
        'rot90_3',  # 旋转270度
        'flip_h',  # 水平翻转
        'flip_v',  # 垂直翻转
        'combo'  # 组合旋转+翻转
    ])

    def apply_aug(img):
        # 应用基础变换
        if choice.startswith('rot90'):
            k = int(choice[-1])
            img = np.rot90(img, k=k, axes=(1, 2))  # 在H,W维度旋转
        elif choice == 'flip_h':
            img = np.flip(img, axis=2)  # 沿W轴翻转
        elif choice == 'flip_v':
            img = np.flip(img, axis=1)  # 沿H轴翻转

        # 组合增强：50%概率额外添加翻转
        if choice == 'combo':
            img = np.rot90(img, k=random.randint(1, 3), axes=(1, 2))
            if random.random() > 0.5:
                img = np.flip(img, axis=1 if random.random() > 0.5 else 2)

        return img.copy()

    return [apply_aug(img) for img in images_list]

class SequentialTransform:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, data):
        for transform in self.transform_list:
            data = transform(data)
        return data

class ApplyToX:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inputs):
        x, y = inputs
        return self.transform(x), y

class ApplyToY:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inputs):
        x, y = inputs
        return x, self.transform(y)

def simple_deg_simulation(img, kernels):
    """
    Pipeline to add synthetic degradations to a (RAW/RGB) image.
    y = down(x * k) + n
    """

    img = convert_to_tensor(img)

    # Apply psf blur: x * k
    img = add_blur(img, kernels)

    # Apply downsampling down(x*k)
    img = downsample_raw(img)

    # Add noise down(x*k) + n
    p_noise = np.random.rand()
    if p_noise > 0.3:
        img = add_natural_noise(img)
    else:
        img = add_heteroscedastic_gnoise(img)

    return img


class DegradationSimulator:
    """
    实现退化模拟的类，执行顺序：模糊 → 下采样 → 加噪声
    """
    def __init__(self, noise_switch_threshold=0.3,
                 blur=True, downsample=True, noise=True):
        """
        初始化模拟器
        :param noise_switch_threshold: 噪声类型切换阈值 (默认0.3，>0.3时加自然噪声)
        """
        self.noise_switch_threshold = noise_switch_threshold
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kernels = np.load(os.path.join(current_dir, 'kernels.npy'), allow_pickle=True)
        self.kernels = np.stack(kernels)
        self.blur = blur
        self.downsample = downsample
        self.noise = noise

    def __call__(self, img):
        """
        执行完整的退化模拟流程
        :param img: 输入图像 (RAW/RGB)
        :param kernels: 模糊核
        :return: 退化后的图像
        """
        # print(img.shape)
        img = img.transpose((1, 2, 0))
        img = self._convert_to_tensor(img)
        kernels = self.kernels
        if self.blur:
            img = self._apply_psf_blur(img, kernels)
        if self.downsample:
            img = self._apply_downsampling(img)
        if self.noise:
            img = self._add_noise_distortion(img)
        img = img.numpy().transpose((2, 0, 1))
        return img

    def _convert_to_tensor(self, img):
        """将图像转换为张量格式"""
        return convert_to_tensor(img)  # 假设已存在该函数

    def _apply_psf_blur(self, img, kernels):
        """应用PSF模糊"""
        return add_blur(img, kernels)  # 假设已存在该函数

    def _apply_downsampling(self, img):
        """执行下采样"""
        return downsample_raw(img)  # 假设已存在该函数

    def _add_noise_distortion(self, img):
        """添加噪声扰动"""
        p_noise = np.random.rand()
        if p_noise > self.noise_switch_threshold:
            return add_natural_noise(img)  # 假设已存在该函数
        return add_heteroscedastic_gnoise(img)  # 假设已存在该函数