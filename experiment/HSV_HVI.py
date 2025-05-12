import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 路径设置
img_dir = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/high'
output_dir = 'HSV_HVI_colorspace'
os.makedirs(output_dir, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))

for img_path in img_paths:
    if not img_path.endswith('492.png'):
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB

    # 获取像素
    h_img, w_img, _ = img.shape
    pixels = img.reshape((-1, 3)).astype(np.float32) / 255.0  # 归一化到0-1

    # 手动RGB转HSV
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    img_max = np.maximum.reduce([r, g, b])
    img_min = np.minimum.reduce([r, g, b])
    delta = img_max - img_min
    eps = 1e-10

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
    # h = h / 6.0  # 归一化到 [0,1]

    # --------------------------------------
    # HVI空间：极化 Polarization
    # --------------------------------------
    H_rad = (np.pi / 3) * h  # H_rad在[0,2pi]

    h_polar = np.cos(H_rad)  # h_polar在[-1,1]
    v_polar = np.sin(H_rad)  # v_polar在[-1,1]

    # (c) HVI极化（未亮度塌缩）
    x_hvi_polar = s * h_polar  # s相当于是圆的半径，s在[0,1]之间
    y_hvi_polar = s * v_polar
    z_hvi_polar = v

    # --------------------------------------
    # HVI空间：亮度塌缩 Collapse
    # --------------------------------------
    k = 1.0
    epsilon = 1e-8
    Ck = k * np.sin(np.pi * v / 2) + epsilon  # 亮度塌缩公式

    x_hvi_collapse = Ck * s * h_polar
    y_hvi_collapse = Ck * s * v_polar
    z_hvi_collapse = Ck

    # --------------------------------------
    # 画图
    # --------------------------------------
    fig = plt.figure(figsize=(12, 6))

    # (c) Polarization only
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_hvi_polar, y_hvi_polar, z_hvi_polar, c=pixels, marker='.', s=0.1)
    ax1.set_xlabel('h * S')
    ax1.set_ylabel('v * S')
    ax1.set_zlabel('V')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(0, 1)
    ax1.set_title('HVI Polarized (Before Collapse)')

    # (d) Polarization + Collapse
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_hvi_collapse, y_hvi_collapse, z_hvi_collapse, c=pixels, marker='.', s=0.1)
    ax2.set_xlabel('h * S * Ck')
    ax2.set_ylabel('v * S * Ck')
    ax2.set_zlabel('Ck')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(0, 1)
    ax2.set_title('HVI Polarized + Collapsed')

    plt.tight_layout()

    # 保存图像
    filename = os.path.basename(img_path).replace('.png', '_HVI.png')
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved HVI visualization: {save_path}")
