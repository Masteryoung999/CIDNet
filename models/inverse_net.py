import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积:先逐通道卷积,再1x1卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class AffineCoupling(nn.Module):
    """
    仿射耦合层: B_out = B * exp(s(A)) + t(A)
    """
    def __init__(self, num_channels):
        super().__init__()
        self.half = num_channels // 2
        self.net = nn.Sequential(
            DepthwiseSeparableConv(self.half, 32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, self.half * 2),
        )

    def forward(self, x, reverse=False):
        A, B = x[:, :self.half], x[:, self.half:]
        st = self.net(A)
        s, t = st[:, :self.half], st[:, self.half:]
        if not reverse:
            B_out = B * torch.exp(s) + t
            out = torch.cat([A, B_out], dim=1)
        else:
            B_rec = (B - t) * torch.exp(-s)
            out = torch.cat([A, B_rec], dim=1)
        return out


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_init = torch.linalg.qr(torch.randn(num_channels, num_channels), mode='reduced')[0]
        self.weight = nn.Parameter(w_init.view(num_channels, num_channels, 1, 1))

    def forward(self, x, reverse=False):
        if not reverse:
            return F.conv2d(x, self.weight)
        else:
            w_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
            return F.conv2d(x, w_inv)


class InvertibleColorTransform(nn.Module):
    """
    将 RGB <-> (L,C1,C2) 的可逆变换
    """
    def __init__(self, num_coupling_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        C = 3
        for _ in range(num_coupling_layers):
            self.layers.append(Invertible1x1Conv(C))
            self.layers.append(AffineCoupling(C))

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z, reverse=False)
        return z

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        return x


if __name__ == "__main__":
    net = InvertibleColorTransform(num_coupling_layers=4)
    img = torch.randn(2, 3, 128, 128)
    z = net(img)
    rec = net.inverse(z)

    print("输入 shape:", img.shape)
    print("输出 shape:", z.shape)
    print("输入与重建的最大误差:", (img - rec).abs().max().item())

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型总参数量:{total_params / 1e6:.3f} M")
    print(f"其中可训练参数量:{trainable_params / 1e6:.3f} M")
