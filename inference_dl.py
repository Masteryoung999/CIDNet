import argparse
import os
import torch
from glob import glob
from setup_dl import Engine
from utility import seed_everywhere, load_img
from options import train_options
import torchvision.transforms.functional as TF


if __name__ == '__main__':

    """Testing settings"""
    parser = argparse.ArgumentParser(description='LLIE')
    opt = train_options(parser)
    opt.no_log = True
    opt.no_ropt = True

    opt.gpu_ids = [7]
    opt.arch = 'CIDNet'
    opt.resume = True
    opt.resumePath = '/data3/yyh/HVI_CIDNet_new/checkpoints/CIDNet/LOLv1/model_best.pth'
    opt.loss = 'CIDLoss'

    print(opt)

    """Set Random Status"""
    seed_everywhere(opt.seed)

    """Setup Engine"""
    engine = Engine(opt)
    print('model params: %.2fM' % (sum([t.nelement() for t in engine.net.parameters()]) / 10 ** 6))

    """Inference"""
    save_dir = os.path.join('results', opt.arch)
    os.makedirs(save_dir, exist_ok=True)

    input_images = sorted(glob("/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/low/*.png"))
    for img_path in input_images:
        # 加载图像并转换为 tensor
        img = load_img(img_path)
        img_tensor = TF.to_tensor(img).unsqueeze(0).to(engine.device)  # [1, C, H, W]

        with torch.no_grad():
            output = engine.net(img_tensor)
            # print(output.shape)
        
        # 如果模型输出是 list 或 tuple，取第一个
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = output.squeeze(0).cpu().clamp(0, 1)  # [C, H, W]

        # 保存图像
        output_img_path = os.path.join(save_dir, os.path.basename(img_path))
        img = TF.to_pil_image(output)
        img.save(output_img_path)

    print(f"Inference complete. Results saved to: {save_dir}")
