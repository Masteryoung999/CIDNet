import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, glob
from matplotlib.colors import hsv_to_rgb

# 输入、输出目录
img_dir = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/high'
out_dir = 'RGB_YCrCb_features_colored'
os.makedirs(out_dir, exist_ok=True)

for path in sorted(glob.glob(os.path.join(img_dir, '*.png'))):
    # 只示例特定图片可注释掉以下两行
    # if not path.endswith('492.png'):
    #     continue

    # 读取图像并分解
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # 展平像素
    h, w, _ = rgb.shape
    rgb_pts = rgb.reshape(-1, 3) / 255.0  # RGB 着色
    Crf = Cr.reshape(-1).astype(np.float32)
    Cbf = Cb.reshape(-1).astype(np.float32)
    Yf = Y.reshape(-1).astype(np.float32)

    # 计算 HSI for color features
    '''
    思路:
    对于每个像素，取 Cr, Cb 分量,计算它们相对于中心(128)的偏移向量
    用该向量的 极角(atan2)作为 HSV 空间的色调(Hue),可视为“色彩方向”
    用该向量的 模长(√((Cr-128)²+(Cb-128)²))归一化后作为饱和度(Saturation),可视为“色彩强度”
    用 Y 分量归一化(Y/255)作为明度(Value),可视为“亮度”
    最后再把 HSV 转回 RGB 着色，这样一个点的颜色同时编码了它的色彩偏向、色彩强度和亮度
    '''
    dCr = Crf - 128.0
    dCb = Cbf - 128.0
    H = (np.arctan2(dCb, dCr) / (2*np.pi)) % 1.0
    S = np.sqrt(dCr**2 + dCb**2) / (np.sqrt(2*(127.0**2)))
    S = np.clip(S, 0, 1)
    V = Yf / 255.0
    hsv = np.stack([H, S, V], axis=1)
    feature_rgb = hsv_to_rgb(hsv)

    # 绘图
    fig = plt.figure(figsize=(12, 6))

    # RGB 空间
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(rgb_pts[:, 0]*255, rgb_pts[:, 1]*255, rgb_pts[:, 2]*255,
                c=rgb_pts, s=0.5, marker='.')
    ax1.set_xlabel('R'); ax1.set_ylabel('G'); ax1.set_zlabel('B')
    ax1.set_xlim(0,255); ax1.set_ylim(0,255); ax1.set_zlim(0,255)
    ax1.set_title('sRGB Color Space')

    # YCrCb 特征空间
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(Crf, Cbf, Yf,
                c=feature_rgb, s=0.5, marker='.')
    ax2.set_xlabel('Cr'); ax2.set_ylabel('Cb'); ax2.set_zlabel('Y')
    ax2.set_xlim(0,255); ax2.set_ylim(0,255); ax2.set_zlim(0,255)
    ax2.set_title('YCrCb Feature Coloring')

    plt.tight_layout()
    save_name = os.path.basename(path).replace('.png','_RGB+YCrCb.png')
    plt.savefig(os.path.join(out_dir, save_name), dpi=200)
    plt.close()

    print(f"Saved: {save_name}")
