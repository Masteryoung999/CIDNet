import argparse
from glob import glob

import numpy as np
from thop import profile
from setup_dl import Engine, train_options
from utility import *
import time

if __name__ == '__main__':

    """Testing settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    opt.no_log = True
    opt.no_ropt = True

    opt.gpu_ids = [7]
    opt.arch = 'mffssr_msca'
    # opt.arch = 'mffssr_tiny_v1'
    # opt.arch = 'rfdn'

    # opt.resume = True
    # opt.resumePath = f'checkpoints/{opt.arch}/NTIRE_rbsformer/model_latest.pth'
    # opt.resumePath = f'checkpoints/{opt.arch}/NTIRE_l1/model_latest.pth'

    opt.loss = 'l1'
    print(opt)

    """Set Random Status"""
    seed_everywhere(opt.seed)

    """Setup Engine"""
    engine = Engine(opt)
    print('model params: %.4f' % (sum([t.nelement() for t in engine.net.parameters()]) / 10 ** 6))

    flops, params = profile(engine.net, inputs=(torch.randn(1, 4, 512, 512).cuda(),))
    print(f"FLOPs: {flops / 1e9} GFLOPs")  # 转换为 GFLOPs
    print(f"Params: {params / 1e6} M")

    """Inference"""
    save_dir = os.path.join('results', opt.arch)
    os.makedirs(save_dir, exist_ok=True)

    time_cost = []
    for rawf in sorted(glob("/data01/pl/HSITask/data/NTIRE_RAWSR2024/val_in/*.npz")):
        raw = np.load(rawf)
        raw_img = raw["raw"]
        raw_max = raw["max_val"]

        raw_img_tensor = torch.from_numpy(raw_img / raw_max).permute(2, 0, 1).unsqueeze(0)
        raw_img_tensor = raw_img_tensor.to(engine.device).float()

        t_s = time.time()
        with torch.no_grad():
            outputs = engine.forward(raw_img_tensor)
        t_e = time.time()
        time_cost.append(t_e - t_s)
    print(time_cost)
    print('time cost: ', np.mean(time_cost), time_cost[-1])







