import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import torch
from math import pi

# 创建输出文件夹
input_dir = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/low'
save_dir = 'RGB_HSV_YCrCb_HVI_4x3_visualization'
os.makedirs(save_dir, exist_ok=True)

# 获取所有图片路径
image_paths = sorted(glob(os.path.join(input_dir, '*.png')))

# 定义 HVIT 函数（从类中抽出来）
def compute_HVI(img_tensor, k=1):
    eps = 1e-8
    device = img_tensor.device
    dtype = img_tensor.dtype

    B, C, H_, W_ = img_tensor.shape
    hue = torch.zeros((B, H_, W_), device=device, dtype=dtype)
    value, _ = img_tensor.max(1)
    img_min, _ = img_tensor.min(1)

    mask_r = img_tensor[:, 0] == value
    mask_g = img_tensor[:, 1] == value
    mask_b = img_tensor[:, 2] == value

    hue[mask_b] = 4.0 + ((img_tensor[:, 0] - img_tensor[:, 1]) / (value - img_min + eps))[mask_b]
    hue[mask_g] = 2.0 + ((img_tensor[:, 2] - img_tensor[:, 0]) / (value - img_min + eps))[mask_g]
    hue[mask_r] = (0.0 + ((img_tensor[:, 1] - img_tensor[:, 2]) / (value - img_min + eps))[mask_r]) % 6
    hue[img_min == value] = 0.0
    hue = hue / 6.0

    saturation = (value - img_min) / (value + eps)
    saturation[value == 0] = 0

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)

    color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
    ch = (2.0 * pi * hue).cos()
    cv = (2.0 * pi * hue).sin()
    H = color_sensitive * saturation * ch
    V = color_sensitive * saturation * cv
    I = value

    return torch.cat([H, V, I], dim=1)


# 遍历图像
for img_path in image_paths:
    filename = os.path.splitext(os.path.basename(img_path))[0]

    # 读取图像
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"无法读取图像：{img_path}")
        continue

    # RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)

    # HSV
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    H_hsv, S_hsv, V_hsv = cv2.split(image_hsv)

    # YCrCb
    image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(image_ycrcb)

    # 计算 HVI
    img_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # shape: [1,3,H,W]
    img_tensor = img_tensor.to('cpu')  # 可换为 'cuda' 取决于你的环境
    HVI_tensor = compute_HVI(img_tensor)[0].detach().cpu().numpy()  # shape: [3,H,W]
    H_hvi, V_hvi, I_hvi = HVI_tensor

    # 可视化：4行 × 3列
    plt.figure(figsize=(12, 16))

    titles = [
        'R Channel', 'G Channel', 'B Channel',
        'H (HSV)', 'S (HSV)', 'V (HSV)',
        'Y (Luminance)', 'Cr', 'Cb',
        'H (HVI)', 'V (HVI)', 'I (HVI)'
    ]
    images = [R, G, B, H_hsv, S_hsv, V_hsv, Y, Cr, Cb, H_hvi, V_hvi, I_hvi]

    for i in range(12):
        plt.subplot(4, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}_4x3_channels.png'), dpi=300)
    plt.close()

    print(f"Saved: {filename}_4x3_channels.png")

print("全部图像处理与保存完成。")
