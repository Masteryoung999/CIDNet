import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 读取图像
img_dir = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/high'
output_dir = 'HSV_RGB_colorspace_high'

os.makedirs(output_dir, exist_ok=True)
img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))

for img_path in img_paths:
    img = cv2.imread(img_path)  # 换成你的图像路径
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 读取的是BGR，转换为RGB

    # 获取图像尺寸
    h, w, _ = img.shape
    pixels = img.reshape((-1, 3))

    # ----------------------------
    # 1. sRGB空间中的点云分布图
    # ----------------------------
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')  # 在1行2列的网格中，当前子图是第1个位置（即左图）
    # 使用圆形 ('o') 作为点的标记，其他可选值：'.'（小点）、's'（方形）、'+'（十字）等，点大小：s=0.1
    ax1.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=pixels / 255.0, marker='.', s=0.1)
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')
    ax1.set_xlim(0, 255)
    ax1.set_ylim(0, 255)
    ax1.set_zlim(0, 255)
    ax1.set_title('sRGB Color Space')

    # ----------------------------
    # 2. HSV空间中的点云分布图
    # ----------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv.reshape((-1, 3))

    # 转换为0-1归一化用于着色
    hsv_norm = hsv_pixels.astype(np.float32)
    hsv_norm[:, 0] = hsv_norm[:, 0] / 180.0  # H是0-180
    hsv_norm[:, 1:] = hsv_norm[:, 1:] / 255.0

    hsv_color = cv2.cvtColor(hsv.reshape((h*w,1,3)), cv2.COLOR_HSV2RGB).reshape((-1,3)) / 255.0

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2], c=hsv_color, marker='.', s=0.1)
    ax2.set_xlabel('H')
    ax2.set_ylabel('S')
    ax2.set_zlabel('V')
    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, 255)
    ax2.set_zlim(0, 255)
    ax2.set_title('HSV Color Space')

    plt.tight_layout()

    # 保存图像
    filename = os.path.basename(img_path).replace('.png', '_colorspace.png')
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved color space visualization: {save_path}")

