import torchvision.utils as vutils
from PIL import Image
import torch.optim as optim
import torch
import torch.nn.functional as F
from models.learnable_colospace import InvertibleColorTransform
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
import numpy as np
import random
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms as t
import matplotlib.pyplot as plt
import os

# === Utility Functions ===
def save_z_channels(z, save_path, index):
    os.makedirs(save_path, exist_ok=True)
    channels = []
    for i in range(3):
        channel_img = z[0, i:i+1]
        grid = vutils.make_grid(channel_img, normalize=True, scale_each=True)
        channels.append(grid)
    combined = torch.cat(channels, dim=2)  # 横向拼接，[C, H, 3*W]
    vutils.save_image(combined, f"{save_path}/z_combined_img_{index}.png")


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_all_files(path):
    path = Path(path).resolve()
    if not path.exists():
        return []
    if not path.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {path}")
    return [str(file_path) for file_path in path.rglob("*") if file_path.is_file() and file_path.name.endswith(".png")]

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

# === Dataset ===
class BaseDataset(data.Dataset):
    def __init__(self, filepath, repeat=1):
        self.length = len(filepath)
        self.filepath = filepath
        if not self.filepath:
            raise ValueError("filepaths 列表为空，请检查数据路径和加载逻辑")
        self.repeat = repeat
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        return self.length * self.repeat

class PairedNPZDataset(BaseDataset):
    def __init__(self, filepath, repeat=1, transform=None):
        super(PairedNPZDataset, self).__init__(filepath, repeat)
        self.filepath = filepath
        self.transform = transform

    def __getitem__(self, index):
        index = index % len(self.filepath)
        fp_in, fp_gt = self.filepath[index]
        img = load_img(fp_in)
        img_gt = load_img(fp_gt)
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(img)
            random.seed(seed)
            torch.manual_seed(seed)
            img_gt = self.transform(img_gt)
        return img, img_gt

# === Losses ===
def L_recon(x, x_rec):
    return F.mse_loss(x_rec, x)

def L_luminance(z, x):
    z_l = z[:, 0:1]
    x_l = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return F.mse_loss(z_l, x_l)

class ProjectionHead(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32, proj_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(x, dim=1)

def info_nce_loss(anchor, positive, negatives, temperature=0.1):
    B = anchor.size(0)
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)

    pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)  # [B, 1]
    neg_sim = torch.matmul(anchor, negatives.T)  # [B, B]

    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, B+1]
    labels = torch.zeros(B, dtype=torch.long).to(anchor.device)

    logits /= temperature
    return F.cross_entropy(logits, labels)

def L_InfoNCE(z, proj_net):
    z_l = z[:, 0:1]
    z_c1 = z[:, 1:2]
    z_c2 = z[:, 2:3]

    z_l_vec = proj_net(z_l)
    z_c1_vec = proj_net(z_c1)
    z_c2_vec = proj_net(z_c2)

    loss_l_c1 = info_nce_loss(z_l_vec, z_l_vec, z_c1_vec)
    loss_l_c2 = info_nce_loss(z_l_vec, z_l_vec, z_c2_vec)
    # loss_c1_c2 = info_nce_loss(z_c1_vec, z_c1_vec, z_c2_vec)

    # return (loss_l_c1 + loss_l_c2 + loss_c1_c2) / 3
    return (loss_l_c1 + loss_l_c2) / 2

# === Data ===
def make_dataset_common(transform, batch_size=None, repeat=1, phase='train', shuffle=True):
    if dataname == 'LOLv1':
        if phase == 'train':
            input_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/low'
            gt_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/our485/low'
        elif phase == 'val':
            input_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/low'
            gt_root = '/data3/yyh/HVI_CIDNet_new/datasets/LOLdataset/eval15/low'
        else:
            raise ValueError(f"无效 phase: {phase}")
        input_files = sorted(get_all_files(input_root))
        gt_files = sorted(get_all_files(gt_root))
        file_paths = list(zip(input_files, gt_files))
        dataset = PairedNPZDataset(file_paths, repeat=repeat, transform=transform)
    else:
        raise ValueError(f"未知数据集: {dataname}")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=threads, pin_memory=False, worker_init_fn=worker_init_fn)
    return data_loader

@torch.no_grad()
def validate(net, val_loader, proj_net, vis_dir='visuals/LCCcolorspace_v2'):
    net.eval()
    os.makedirs(vis_dir, exist_ok=True)
    val_loss = 0
    val_recon = 0
    val_lum = 0
    val_infonce = 0
    index = 0
    for x, _ in val_loader:
        index =  index + 1
        x = x.to(device)
        z = net(x)
        x_rec = net.inverse(z)

        save_z_channels(z, vis_dir, index)

        loss_recon = L_recon(x, x_rec)
        loss_lum = L_luminance(z, x)
        loss_infonce = L_InfoNCE(z, proj_net)
        loss = loss_recon + lambda_lum * loss_lum + lambda_infonce * loss_infonce

        val_loss += loss.item()
        val_recon += loss_recon.item()
        val_lum += loss_lum.item()
        val_infonce += loss_infonce.item()
    n = len(val_loader)
    print(f"[Val] Recon: {val_recon / n:.6f}, Lum: {val_lum / n:.6f}, InfoNCE: {val_infonce / n:.6f}, Total: {val_loss / n:.6f}")
    return val_loss / n

# === Training Params ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-4
lambda_lum = 100
lambda_infonce = 1.0
dataname = 'LOLv1'
cropSize = 400
batchSize = 6
epochs = 1000
threads = 8
prefix = f'{dataname}_colorspace_v2'
gpu_ids = [0]
seed = 2025

seed_everywhere(seed)
torch.cuda.set_device(gpu_ids[0])

train_loader = make_dataset_common(Compose([RandomCrop((cropSize, cropSize)), RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()]),
                                   batchSize, repeat=4, shuffle=True, phase='train')
val_loader = make_dataset_common(Compose([ToTensor()]), 1, repeat=1, shuffle=False, phase='val')

net = InvertibleColorTransform(num_coupling_layers=4).to(device)
proj_net = ProjectionHead().to(device)
optimizer = optim.Adam(list(net.parameters()) + list(proj_net.parameters()), lr=lr)

best_loss = float('inf')
os.makedirs(f'checkpoints/{prefix}', exist_ok=True)

for epoch in range(epochs):
    net.train()
    total_loss = 0
    total_recon = 0
    total_lum = 0
    total_infonce = 0

    for x, _ in train_loader:
        x = x.to(device)
        z = net(x)
        x_rec = net.inverse(z)
        loss_recon = L_recon(x, x_rec)
        loss_lum = L_luminance(z, x)
        loss_infonce = L_InfoNCE(z, proj_net)
        loss = loss_recon + lambda_lum * loss_lum + lambda_infonce * loss_infonce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_lum += loss_lum.item()
        total_infonce += loss_infonce.item()

    n = len(train_loader)
    val_loss = validate(net, val_loader, proj_net)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(net.state_dict(), f'checkpoints/{prefix}/model_best.pth')

    torch.save(net.state_dict(), f'checkpoints/{prefix}/model_latest.pth')

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Recon: {total_recon / n:.6f}, "
          f"Train Lum: {total_lum / n:.6f}, "
          f"Train InfoNCE: {total_infonce / n:.6f}, "
          f"Train Total: {total_loss / n:.6f}, "
          f"Val Total: {val_loss:.6f}")
