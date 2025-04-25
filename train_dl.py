import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from setup_dl import Engine
from utility import seed_everywhere, display_learning_rate, \
    make_dataset_common, load_datasets, SequentialTransform, \
    crop_rand, data_augment, to_tensor_transform
from options.options import train_options


if __name__ == '__main__':
    torch.set_num_threads(4)
    """Testing settings"""
    parser = argparse.ArgumentParser(
        description='LLIE')
    opt = train_options(parser)
    opt.gpu_ids = [6]
    opt.arch = 'CIDNet'
    opt.dataname = 'LOLv1'
    
    opt.loss = 'CIDLoss'
    # opt.prefix = f'{opt.dataname}'
    opt.prefix = f'{opt.dataname}_{opt.loss}'

    # opt.resume = True
    # opt.resumePath = f'checkpoints/mffssr_large/NTIRE_rbsformer/model_latest.pth'
    # opt.clear = True
    # opt.no_ropt = True
    print(opt)

    """Set Random Status"""
    seed_everywhere(opt.seed)

    
    # train_transform = SequentialTransform(
    #     [
    #         lambda data: crop_rand(data, 400, 400),
    #         lambda data: data_augment(data),
    #         lambda data: [t.transpose(1,2,0) for t in data],
    #         lambda data: to_tensor_transform(data),
    #         # norm：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     ]
    # )
    # val_transform = SequentialTransform(
    #     [
    #         lambda data: to_tensor_transform(data)
    #     ]
    # )

    # repeat = 1
 
    # trainset = make_dataset_common(opt, train_transform, batch_size,
    #                 repeat=repeat, shuffle=True, phase='train')
    # valset = make_dataset_common(opt, val_transform, 1,
    #                 repeat=1, shuffle=False, phase='val')
    
    trainset, valset = load_datasets()

    opt.iterations = opt.epoch * len(trainset)
    print(f'total iterations: {opt.iterations}')
    engine = Engine(opt)
    print('model params: %.4fM' % (sum([t.nelement() for t in engine.net.parameters()]) / 10 ** 6))

    epoch_per_save = max(opt.epoch // 10, 1)
    rand_status_list = np.random.get_state()[1].tolist()
    while engine.epoch < opt.epoch:
        np.random.seed(rand_status_list[engine.epoch % len(rand_status_list)])  # reset seed per epoch, otherwise the noise will be added with a specific pattern

        engine.train(trainset)

        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

        avg_psnr, avg_loss = engine.validate(valset, 'validate')
        if avg_psnr > engine.best_psnr:
            engine.best_psnr = avg_psnr
            engine.best_epoch = engine.epoch
            model_best_path = os.path.join(engine.basedir, engine.prefix, 'model_best.pth')
            print('')
            engine.save_checkpoint(
                model_out_path=model_best_path
            )


