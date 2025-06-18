import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 深度卷积（分组卷积）
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        # 逐点卷积（1x1调整通道数）
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, num_channels, cond_dim):
        super().__init__()
        self.half = num_channels // 2
        self.cond_proj = nn.Linear(cond_dim, 128)  # 条件投影层
        
        # 包含条件输入的网络结构
        self.net = nn.Sequential(
            DepthwiseSeparableConv(self.half + 128, 128),  # 增加条件通道
            nn.ReLU(),
            DepthwiseSeparableConv(128, 128),
            nn.ReLU(),
            DepthwiseSeparableConv(128, self.half * 2),
        )

    def forward(self, x, cond, reverse=False):
        A, B = x[:, :self.half], x[:, self.half:]
        
        # 条件处理
        cond = self.cond_proj(cond)
        cond = cond.view(cond.size(0), 128, 1, 1).expand(-1, -1, A.size(2), A.size(3))
        
        # 拼接条件和A部分
        A_cond = torch.cat([A, cond], dim=1)
        
        st = self.net(A_cond)
        s, t = st[:, :self.half], st[:, self.half:]
        s = torch.tanh(s)  # 限制缩放范围
        
        if not reverse:
            B_out = B * torch.exp(s) + t
            return torch.cat([A, B_out], dim=1)
        else:
            B_rec = (B - t) * torch.exp(-s)
            return torch.cat([A, B_rec], dim=1)

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_init = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(w_init.view(num_channels, num_channels, 1, 1))

    def forward(self, x, reverse=False):
        if not reverse:
            return F.conv2d(x, self.weight)
        else:
            w_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
            return F.conv2d(x, w_inv)

class CompressionEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            DepthwiseSeparableConv(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            DepthwiseSeparableConv(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),           
            DepthwiseSeparableConv(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),           
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, latent_dim * 2)

    def forward(self, x):
        features = self.net(x).view(x.size(0), -1)
        params = self.fc(features)
        mu, logvar = torch.chunk(params, 2, dim=1)
        return mu, logvar

class ConditionalFlowDecoder(nn.Module):
    def __init__(self, num_channels=3, cond_dim=128, num_coupling=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_coupling):
            self.layers.append(Invertible1x1Conv(num_channels))
            self.layers.append(ConditionalAffineCoupling(num_channels, cond_dim))

    def forward(self, x, cond):
        for layer in self.layers:
            if isinstance(layer, ConditionalAffineCoupling):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x

    def inverse(self, z, cond):
        x = z
        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalAffineCoupling):
                x = layer(x, cond, reverse=True)
            else:
                x = layer(x, reverse=True)
        return x

class VAEWithConditionalFlow(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = CompressionEncoder(latent_dim=latent_dim)
        self.decoder = ConditionalFlowDecoder(cond_dim=latent_dim)
        
        # 先验分布参数
        self.prior_mu = nn.Parameter(torch.zeros(1))
        self.prior_std = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 编码阶段
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z)
        
        # 解码阶段
        v = self.decoder(x, z)  # 前向变换
        log_prob = self._calculate_log_prob(v)
        
        # 计算损失
        recon_loss = -log_prob.mean()
        kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        
        return {'loss': recon_loss + kl_loss, 
                'recon_loss': recon_loss,
                'kl_loss': kl_loss}

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _calculate_log_prob(self, v):
        # 假设v服从标准正态分布
        return torch.sum(Normal(0, 1).log_prob(v), dim=[1,2,3])

    def generate(self, bs, device):
        # 从先验采样
        z = torch.randn(bs, 128).to(device)
        v = torch.randn(bs, 3, 128, 128).to(device)
        return self.decoder.inverse(v, z)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAEWithConditionalFlow().to(device)
    
    # 测试前向传播
    img = torch.randn(2, 3, 128, 128).to(device)
    losses = model(img)
    print(f"总损失: {losses['loss'].item():.3f}")
    print(f"重构损失: {losses['recon_loss'].item():.3f}")
    print(f"KL损失: {losses['kl_loss'].item():.3f}")

    # 测试生成
    with torch.no_grad():
        gen_img = model.generate(2, device)
        print("生成图像形状:", gen_img.shape)
    print("输入与重建的最大误差:", (img - gen_img).abs().max().item())
    # 参数量统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total/1e6:.2f}M")
    print(f"可训练参数: {trainable/1e6:.2f}M")