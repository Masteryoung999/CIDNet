import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from glob import glob

# ========= 参数设置 =========
img_dir = 'datasets/LOLdataset/eval15/high'
output_dir = 'black_red_noise_figures_high'
mode = 'red'  # 可选：'all' | 'red' | 'black'

# ============================
os.makedirs(output_dir, exist_ok=True)
img_paths = sorted(glob(os.path.join(img_dir, '*.png')))

for img_path in img_paths:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 标准化 HSV
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    hsv_flat = hsv.reshape(-1, 3)

    # 为了上色，将 HSV 还原
    hsv_for_color = hsv.copy()
    hsv_for_color[:, :, 0] *= 180
    hsv_for_color[:, :, 1:] *= 255
    hsv_for_color = hsv_for_color.astype(np.uint8)
    hsv_rgb_color = cv2.cvtColor(hsv_for_color, cv2.COLOR_HSV2RGB).reshape(-1, 3) / 255.0

    # 定义掩码
    red_mask = (hsv_flat[:, 0] < 0.15) | (hsv_flat[:, 0] > 0.85)
    black_mask = hsv_flat[:, 2] < 0.02

    # 创建 3D 图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if mode == 'all':
        ax.scatter(hsv_flat[:, 0], hsv_flat[:, 1], hsv_flat[:, 2],
                   c=hsv_rgb_color, marker='.', s=0.1, alpha=0.3)

        ax.scatter(hsv_flat[red_mask, 0], hsv_flat[red_mask, 1], hsv_flat[red_mask, 2],
                   c='red', marker='.', s=0.5, alpha=0.6, label='Red Discontinuity')

        ax.scatter(hsv_flat[black_mask, 0], hsv_flat[black_mask, 1], hsv_flat[black_mask, 2],
                   c='black', marker='.', s=0.5, alpha=0.6, label='Black Plane')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title('HSV Color Space - All')

    elif mode == 'red':
        ax.scatter(hsv_flat[red_mask, 0], hsv_flat[red_mask, 1], hsv_flat[red_mask, 2],
                   c='red', marker='.', s=0.5, alpha=0.6, label='Red Discontinuity')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title('HSV Red Noise Only')

    elif mode == 'black':
        ax.scatter(hsv_flat[black_mask, 0], hsv_flat[black_mask, 1], hsv_flat[black_mask, 2],
                   c='black', marker='.', s=0.5, alpha=0.6, label='Black Plane')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title('HSV Black Noise Only')

    else:
        raise ValueError("mode 只能是 'all', 'red', 或 'black'")

    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')
    ax.legend()

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f'{base_name}_hsv_{mode}_noise.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")
