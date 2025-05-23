import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# 输入和输出路径
input_dir = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/high'  # 替换为你的图像文件夹路径
save_dir = 'HSV_visualization'
os.makedirs(save_dir, exist_ok=True)

# 获取所有图片路径（假设为 .png 格式，也可以加 jpg 等）
image_paths = sorted(glob(os.path.join(input_dir, '*.png')))

# 处理每张图像
for img_path in image_paths:
    filename = os.path.splitext(os.path.basename(img_path))[0]

    # 读取图像
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"无法读取图像：{img_path}")
        continue

    # 转换为RGB用于显示
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 转换为HSV并拆分通道
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(image_hsv)

    # 保存每个分量为灰度图像
    cv2.imwrite(os.path.join(save_dir, f'{filename}_H.png'), H)
    cv2.imwrite(os.path.join(save_dir, f'{filename}_S.png'), S)
    cv2.imwrite(os.path.join(save_dir, f'{filename}_V.png'), V)

    # 可视化保存
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title('Original RGB')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('H (Hue)')
    # Hue 通道值在 0-180 范围内，直接用灰度可视化
    plt.imshow(H, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('S (Saturation)')
    plt.imshow(S, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('V (Value)')
    plt.imshow(V, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}_HSV_vis.png'))
    plt.close()

print("全部处理完成。")
