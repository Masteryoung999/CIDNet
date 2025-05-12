import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 路径设置
# img_dir = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/low'
img_dir = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/high'
output_dir = 'HSV_RGB_colorspace_trainset'
os.makedirs(output_dir, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))


for img_path in img_paths:
    if  not img_path.endswith('687.png'):
        continue
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB

    # 获取像素
    h, w, _ = img.shape

    pixels = img.reshape((-1, 3)).astype(np.float32) / 255.0  # 归一化到0-1

    # ----------------------------
    # 1. sRGB空间中的点云分布图
    # ----------------------------
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pixels[:, 0]*255, pixels[:, 1]*255, pixels[:, 2]*255, c=pixels, marker='.', s=0.1)
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')
    ax1.set_xlim(0, 255)
    ax1.set_ylim(0, 255)
    ax1.set_zlim(0, 255)
    ax1.set_title('sRGB Color Space')

    # ----------------------------
    # 2. HSV空间中的点云分布图 (手动)
    # ----------------------------

    # 手动RGB转HSV
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    img_max = np.maximum.reduce([r, g, b])
    img_min = np.minimum.reduce([r, g, b])
    delta = img_max - img_min
    eps = 1e-10  # 防止除零

    v = img_max
    s = delta / (img_max + eps)
    h = np.zeros_like(v)

    mask = delta > eps
    idx = (img_max == r) & mask
    h[idx] = ((g[idx] - b[idx]) / (delta[idx] + eps)) % 6
    idx = (img_max == g) & mask
    h[idx] = ((b[idx] - r[idx]) / (delta[idx] + eps)) + 2
    idx = (img_max == b) & mask
    h[idx] = ((r[idx] - g[idx]) / (delta[idx] + eps)) + 4
    h = h / 6.0  # 归一化到 [0,1]

    hsv_pixels = np.stack([h * 180, s * 255, v * 255], axis=1)  # 保持和OpenCV一致的量纲
    print(hsv_pixels.shape)
    print(hsv_pixels)
    # 手动HSV -> RGB用于着色
    hh = h
    ss = s
    vv = v

    c = ss * vv
    x = c * (1 - np.abs((hh * 6) % 2 - 1))
    m = vv - c

    r2 = np.zeros_like(hh)
    g2 = np.zeros_like(hh)
    b2 = np.zeros_like(hh)

    h6 = hh * 6
    idx = (h6 >= 0) & (h6 < 1)
    r2[idx], g2[idx], b2[idx] = c[idx], x[idx], 0
    idx = (h6 >= 1) & (h6 < 2)
    r2[idx], g2[idx], b2[idx] = x[idx], c[idx], 0
    idx = (h6 >= 2) & (h6 < 3)
    r2[idx], g2[idx], b2[idx] = 0, c[idx], x[idx]
    idx = (h6 >= 3) & (h6 < 4)
    r2[idx], g2[idx], b2[idx] = 0, x[idx], c[idx]
    idx = (h6 >= 4) & (h6 < 5)
    r2[idx], g2[idx], b2[idx] = x[idx], 0, c[idx]
    idx = (h6 >= 5) & (h6 <= 6)
    r2[idx], g2[idx], b2[idx] = c[idx], 0, x[idx]

    r2 = r2 + m
    g2 = g2 + m
    b2 = b2 + m

    hsv_color = np.stack([r2, g2, b2], axis=1)

    # 画HSV点云
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2], c=hsv_color, marker='.', s=0.1)
    ax2.set_xlabel('H')
    ax2.set_ylabel('S')
    ax2.set_zlabel('V')
    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, 255)
    ax2.set_zlim(0, 255)
    ax2.set_title('HSV Color Space (Manual)')

    plt.tight_layout()

    # 保存图像
    filename = os.path.basename(img_path).replace('.png', '_colorspace_manual.png')
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved manual HSV color space visualization: {save_path}")
