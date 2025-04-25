import os
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
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
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip



parser = argparse.ArgumentParser(
    description='LLIE')

opt = train_options(parser)
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])
 
 
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform2():
    return Compose([ToTensor()])



def get_lol_training_set(data_dir,size):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


# def get_lol_v2_training_set(data_dir,size):
#     return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


# def get_training_set_blur(data_dir,size):
#     return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


# def get_lol_v2_syn_training_set(data_dir,size):
#     return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


# def get_SID_training_set(data_dir,size):
#     return SIDDatasetFromFolder(data_dir, transform=transform1(size))


# def get_SICE_training_set(data_dir,size):
#     return SICEDatasetFromFolder(data_dir, transform=transform1(size))

# def get_SICE_eval_set(data_dir):
#     return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())
 
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

 
def load_datasets():
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        # if opt.lol_blur:
        #     train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
        #     training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        #     test_set = get_eval_set(opt.data_val_lol_blur)
        #     testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        # if opt.lolv2_real:
        #     train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
        #     training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        #     test_set = get_eval_set(opt.data_val_lolv2_real)
        #     testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        # if opt.lolv2_syn:
        #     train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
        #     training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        #     test_set = get_eval_set(opt.data_val_lolv2_syn)
        #     testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        # if opt.SID:
        #     train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
        #     training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        #     test_set = get_eval_set(opt.data_val_SID)
        #     testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        # if opt.SICE_mix:
        #     train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        #     training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        #     test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        #     testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        # if opt.SICE_grad:
        #     train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        #     training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        #     test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        #     testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

 
class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/low'
        folder2= self.data_dir+'/high'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        # _, file1 = os.path.split(data_filenames[index])
        # _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            # print(im1.shape)
            # print(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
        return im1, im2

    def __len__(self):
        return 485

    
# class LOLv2DatasetFromFolder(data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(LOLv2DatasetFromFolder, self).__init__()
#         self.data_dir = data_dir
#         self.transform = transform

#     def __getitem__(self, index):

#         folder = self.data_dir+'/Low'
#         folder2= self.data_dir+'/Normal'
#         data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
#         data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        
#         im1 = load_img(data_filenames[index])
#         im2 = load_img(data_filenames2[index])
#         _, file1 = os.path.split(data_filenames[index])
#         _, file2 = os.path.split(data_filenames2[index])
#         seed = random.randint(1, 1000000)
#         seed = np.random.randint(seed) # make a seed with numpy generator 
#         if self.transform:
#             random.seed(seed) # apply this seed to img tranforms
#             torch.manual_seed(seed) # needed for torchvision 0.7
#             im1 = self.transform(im1)      
#             random.seed(seed) # apply this seed to img tranforms
#             torch.manual_seed(seed) # needed for torchvision 0.7 
#             im2 = self.transform(im2)
#         return im1, im2, file1, file2

#     def __len__(self):
#         return 685



# class LOLv2SynDatasetFromFolder(data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(LOLv2SynDatasetFromFolder, self).__init__()
#         self.data_dir = data_dir
#         self.transform = transform
#         self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def __getitem__(self, index):

#         folder = self.data_dir+'/Low'
#         folder2= self.data_dir+'/Normal'
#         data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
#         data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]


#         im1 = load_img(data_filenames[index])
#         im2 = load_img(data_filenames2[index])
#         _, file1 = os.path.split(data_filenames[index])
#         _, file2 = os.path.split(data_filenames2[index])
#         seed = random.randint(1, 1000000)
#         seed = np.random.randint(seed) # make a seed with numpy generator 
#         if self.transform:
#             random.seed(seed) # apply this seed to img tranfsorms
#             torch.manual_seed(seed) # needed for torchvision 0.7
#             im1 = self.transform(im1)
#             random.seed(seed)
#             torch.manual_seed(seed)         
#             im2 = self.transform(im2)
#         return im1, im2, file1, file2

#     def __len__(self):
#         return 900

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        folder = self.data_dir+'/low'
        folder2= self.data_dir+'/high'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])

        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        return im1, im2

    def __len__(self):
        return 15

if __name__ == '__main__':
    pass
