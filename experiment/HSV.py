import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from glob import glob

# 图像文件夹路径
img_dir = '/data3/yyh/HVI_CIDNet_new/results/CIDNet'
output_dir = 'colorspace_figures'
os.makedirs(output_dir, exist_ok=True)

# 遍历所有 PNG 图片
img_paths = sorted(glob(os.path.join(img_dir, '*.png')))

for path in img_paths:
    # 读取图像
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    pixels = img.reshape((-1, 3))

    fig = plt.figure(figsize=(10, 5))

    # -------- sRGB空间 --------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=pixels / 255.0, marker='.', s=0.1)
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')
    ax1.set_title('sRGB Color Space')

    # -------- HSV空间 --------
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv.reshape((-1, 3))

    # 随机采样一些像素
    num_points = min(100000, h * w)
    idx = np.random.choice(h * w, size=num_points, replace=False)
    hsv_sample = hsv_pixels[idx]

    ax2 = fig.add_subplot(122, projection='3d')
    red_mask = (hsv_sample[:, 0] < 10) | (hsv_sample[:, 0] > 170)
    ax2.scatter(hsv_sample[red_mask, 0], hsv_sample[red_mask, 1], hsv_sample[red_mask, 2],
                marker='.', s=0.1, alpha=0.6)
    ax2.set_xlabel('H')
    ax2.set_ylabel('S')
    ax2.set_zlabel('V')
    ax2.set_title('HSV Color Space')

    plt.tight_layout()

    # 保存图像
    filename = os.path.basename(path).replace('.png', '_colorspace.png')
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved color space visualization: {save_path}")
