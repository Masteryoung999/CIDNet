import os
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import random
from scipy.io import loadmat
from .utils import worker_init_fn, get_all_files
from torch.utils.data.distributed import DistributedSampler
 
 
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
    def __getitem__(self, index):
        index = index % len(self.filepath)
        fp_in, fp_gt = self.filepath[index]
        img = load_img(fp_in)
        img_gt = load_img(fp_gt)
        return img, img_gt
 
class ImageTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(ImageTransformDataset, self).__init__()
 
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)
 
    def __len__(self):
        return self.length
 
    def __getitem__(self, idx):
        inputs, targets = self.dataset[idx]
        if self.transform is not None:
            inputs, targets = self.transform((inputs, targets))
        return inputs, targets
 
def make_dataset_common(opt, transform, batch_size=None, repeat=1, phase='train', shuffle=True):
    if opt.dataname == 'LOLv1':
        if phase == 'train':
            input_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/low'
            gt_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/high'
            input_files = sorted(get_all_files(input_root))
            gt_files = sorted(get_all_files(gt_root))
            file_paths = list(zip(input_files, gt_files))
            dataset = PairedNPZDataset(file_paths, repeat=repeat)
        elif phase == 'val':
            input_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/low'
            gt_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/high'
            input_files = sorted(get_all_files(input_root))
            gt_files = sorted(get_all_files(gt_root))
            file_paths = list(zip(input_files, gt_files))
            dataset = PairedNPZDataset(file_paths, repeat=1)
 
    else:
        raise 'no such dataset'
 
    """Split patches dataset into training, validation parts"""
    # dataset = TransformDataset(dataset)
    dataset = ImageTransformDataset(dataset, transform)
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
 
 
if __name__ == '__main__':
    pass
