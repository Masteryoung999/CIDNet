from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import random
from scipy.io import loadmat
from .utils import worker_init_fn, get_all_files
from torch.utils.data.distributed import DistributedSampler
from options.options import train_options
import argparse
from os import listdir
from os.path import join
from torchvision import transforms as t



parser = argparse.ArgumentParser(
    description='LLIE')

opt = train_options(parser)
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])
 
 
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

 
class BaseDataset(data.Dataset):
    def __init__(self, filepath, repeat=1):
        self.length = len(filepath)
        self.filepath = filepath
        if not self.filepath:
            raise ValueError("filepaths 列表为空，请检查数据路径和加载逻辑")
        self.repeat = repeat
 
    def __getitem__(self, index):
        pass
 
    def __len__(self):
        return self.length * self.repeat
 
 
class PairedNPZDataset(BaseDataset):
    def __init__(self, filepath, repeat=1, transform=None):
        super(PairedNPZDataset, self).__init__(filepath, repeat)
        self.filepath = filepath
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # LOLv1 mean std

    def __getitem__(self, index):
        index = index % len(self.filepath)
        fp_in, fp_gt = self.filepath[index]
        img = load_img(fp_in)
        img_gt = load_img(fp_gt)
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            img = self.transform(img)
            random.seed(seed)
            torch.manual_seed(seed)  
            img_gt = self.transform(img_gt)    
        return img, img_gt
    
    def __len__(self):
        return self.length

 
def make_dataset_common(opt, transform, batch_size=None, repeat=1, phase='train', shuffle=True):
    if opt.dataname == 'LOLv1':
        if phase == 'train':
            input_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/low'
            gt_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/high'
            input_files = sorted(get_all_files(input_root))
            gt_files = sorted(get_all_files(gt_root))
            file_paths = list(zip(input_files, gt_files))
            dataset = PairedNPZDataset(file_paths, repeat=repeat, transform=transform)
        elif phase == 'val':
            input_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/low'
            gt_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/high'
            input_files = sorted(get_all_files(input_root))
            gt_files = sorted(get_all_files(gt_root))
            file_paths = list(zip(input_files, gt_files))
            dataset = PairedNPZDataset(file_paths, repeat=1, transform=transform)
 
    else:
        raise 'no such dataset'
 
    """Split patches dataset into training, validation parts"""
    if opt.world_size > 1 and phase == 'train':
        sampler = DistributedSampler(dataset, num_replicas=opt.world_size,
                                      rank=opt.rank, shuffle=shuffle)
        data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                                   batch_size=batch_size,
                                                   num_workers=opt.threads)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                   batch_size=batch_size, shuffle=shuffle,
                                   num_workers=opt.threads, pin_memory=False,
                                   worker_init_fn=worker_init_fn)
 
    return data_loader

 
