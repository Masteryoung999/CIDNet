import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,6'
import argparse
import numpy as np
import matplotlib.pyplot as plt
from setup_dl_dist import Engine
from utility import seed_everywhere, display_learning_rate, \
    make_dataset_common, SequentialTransform, ApplyToX, \
    crop_rand, DegradationSimulator, data_augment, identity
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from options import teacher_options, student_options


def main(rank, world_size):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

    # world_size, rank = 1, 0
    print('world size: ', world_size, 'rank: ', rank)


    """student model settings"""
    torch.set_num_threads(8)
    parser_s = argparse.ArgumentParser(description='super resolution')
    opt_s = student_options(parser_s)
    opt_s.arch = 'mffssr_small'
    opt_s.dataname = 'NTIRE'
    opt_s.loss = 'rbsformer'
    opt_s.prefix = f'{opt_s.dataname}_{opt_s.loss}'
    opt_s.resume = True
    opt_s.resumePath = '/data3/yyh/distillation_v2/checkpoints/mffssr_small/NTIRE_rbsformer/model_latest.pth'
    opt_s.world_size = world_size
    opt_s.rank = rank
    print(opt_s)
    """teacher model settings"""
    parser_t = argparse.ArgumentParser(description='super resolution')
    opt_t = teacher_options(parser_t)
    opt_t.no_log = True
    opt_t.no_ropt = True
    opt_t.arch = 'mffssr_large_final'
    opt_t.resume = True
    opt_t.resumePath = '/data3/yyh/distillation_v2/checkpoints/mffssr_large_final/NTIRE_rbsformer/model_large_final.pth'
    opt_t.world_size = world_size
    opt_t.rank = rank
    print(opt_t)

    """Set Random Status"""
    seed_everywhere(opt_s.seed)
    seed_everywhere(opt_t.seed)

    train_transform = SequentialTransform(
        [
            partial(crop_rand, cropx=256, cropy=256),
            data_augment,
            ApplyToX(DegradationSimulator(blur=True, downsample=True, noise=True))
            ]
    )
    val_transform = identity
    repeat = 1
    batch_size = 1

    trainset = make_dataset_common(opt_s, train_transform, batch_size,
                    repeat=repeat, shuffle=True, phase='train')
    valset = make_dataset_common(opt_s, val_transform, 1,
                    repeat=1, shuffle=False, phase='val')
    opt_s.iterations = opt_s.epoch * len(trainset)
    print(f'total iterations: {opt_s.iterations}')
    engine_t = Engine(opt_t)
    print('model params: %.4f' % (sum([t.nelement() for t in engine_t.net.parameters()]) / 10 ** 6))
    engine_s = Engine(opt_s, engine_teacher = engine_t)
    print('model params: %.4f' % (sum([t.nelement() for t in engine_s.net.parameters()]) / 10 ** 6))
    

    epoch_per_save = max(opt_s.epoch // 10, 1)
    rand_status_list = np.random.get_state()[1].tolist()

    while engine_s.epoch < opt_s.epoch:
        np.random.seed(rand_status_list[engine_s.epoch % len(rand_status_list)])  # reset seed per epoch, otherwise the noise will be added with a specific pattern

        engine_s.train(trainset)

        if rank == 0:
            for param_group in engine_s.optimizer.param_groups:
                print("Current learning rate:", param_group['lr'])

            print('Latest Result Saving...')
            model_latest_path = os.path.join(engine_s.basedir, engine_s.prefix, 'model_latest.pth')
            engine_s.save_checkpoint(
                model_out_path=model_latest_path
            )

        display_learning_rate(engine_s.optimizer)
        if engine_s.epoch % epoch_per_save == 0:
            engine_s.save_checkpoint()

        avg_psnr, avg_loss = engine_s.validate(valset, 'validate')
        if avg_psnr > engine_s.best_psnr:
            engine_s.best_psnr = avg_psnr
            engine_s.best_epoch = engine_s.epoch
            model_best_path = os.path.join(engine_s.basedir, engine_s.prefix, 'model_best.pth')
            print('')
            engine_s.save_checkpoint(
                model_out_path=model_best_path
            )

            if engine_s.opt.scheduler == 'reduce':
                engine_s.scheduler.step(avg_psnr)

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # GPU数量
    mp.spawn(main, args=(world_size,), nprocs=world_size)


