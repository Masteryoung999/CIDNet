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

# class AffineCoupling(nn.Module):
#     """
#     仿射耦合层: B_out = B * exp(s(A)) + t(A)
#     修改为固定使用前两个通道作为A部分
#     """
#     def __init__(self, num_channels):
#         super().__init__()
#         # 固定A部分为2个通道
#         self.A_channels = 2
#         # B部分为1个通道
#         self.B_channels = 1
        
#         self.net = nn.Sequential(
#             DepthwiseSeparableConv(self.A_channels, 64),
#             nn.ReLU(),
#             DepthwiseSeparableConv(64, 64),
#             nn.ReLU(),
#             DepthwiseSeparableConv(64, self.B_channels * 2),
#         )

#     def forward(self, x, reverse=False):
#         # 固定前两个通道为A部分
#         A, B = x[:, :self.A_channels], x[:, self.A_channels:]
        
#         # 生成用于B部分的变换参数
#         st = self.net(A)
#         s, t = st[:, :self.B_channels], st[:, self.B_channels:]
#         s = torch.tanh(s)
        
#         if not reverse:
#             B_out = B * torch.exp(s) + t
#             out = torch.cat([A, B_out], dim=1)
#         else:
#             B_rec = (B - t) * torch.exp(-s)
#             out = torch.cat([A, B_rec], dim=1)
#         return out
class AffineCoupling(nn.Module):
    """
    仿射耦合层: B_out = B * exp(s(A)) + t(A)
    """
    def __init__(self, num_channels):
        super().__init__()
        self.half = num_channels // 2
        self.net = nn.Sequential(
            DepthwiseSeparableConv(self.half, 128),
            nn.ReLU(),
            DepthwiseSeparableConv(128, 128),
            nn.ReLU(),
            DepthwiseSeparableConv(128, self.half * 2),
        )

    def forward(self, x, reverse=False):
        A, B = x[:, :self.half], x[:, self.half:]
        st = self.net(A)
        s, t = st[:, :self.half], st[:, self.half:]
        s = torch.tanh(s)
        if not reverse:
            B_out = B * torch.exp(s) + t
            out = torch.cat([A, B_out], dim=1)
        else:
            B_rec = (B - t) * torch.exp(-s)
            out = torch.cat([A, B_rec], dim=1)
        return out

# class AffineCoupling(nn.Module):
#     """
#     仿射耦合层: B_out = B * exp(s(A)) + t(A)
#     加入通道和空间注意力机制
#     """
#     def __init__(self, num_channels):
#         super().__init__()
#         self.half = num_channels // 2
        
#         # 通道注意力模块
#         self.channel_attention = ChannelAttention(64)
        
#         # 空间注意力模块
#         self.spatial_attention = SpatialAttention()
        
#         self.net = nn.Sequential(
#             DepthwiseSeparableConv(self.half, 64),
#             nn.ReLU(),
#             # 在这里插入通道注意力
#             AttentionWrapper(self.channel_attention),
#             DepthwiseSeparableConv(64, 64),
#             nn.ReLU(),
#             # 在这里插入空间注意力
#             AttentionWrapper(self.spatial_attention),
#             DepthwiseSeparableConv(64, self.half * 2),
#         )

#     def forward(self, x, reverse=False):
#         A, B = x[:, :self.half], x[:, self.half:]
#         st = self.net(A)
#         s, t = st[:, :self.half], st[:, self.half:]
#         s = torch.tanh(s)
#         if not reverse:
#             B_out = B * torch.exp(s) + t
#             out = torch.cat([A, B_out], dim=1)
#         else:
#             B_rec = (B - t) * torch.exp(-s)
#             out = torch.cat([A, B_rec], dim=1)
#         return out


# # 通道注意力模块
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         # 共享MLP
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
#         )
        
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         avg_out = self.mlp(self.avg_pool(x))
#         max_out = self.mlp(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x


# # 空间注意力模块
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         # 沿通道维度计算平均值和最大值
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
        
#         # 拼接特征
#         attention = torch.cat([avg_out, max_out], dim=1)
#         attention = self.conv(attention)
        
#         # 应用空间注意力
#         return self.sigmoid(attention) * x


# # 包装注意力模块使其可以在Sequential中使用
# class AttentionWrapper(nn.Module):
#     def __init__(self, attention_module):
#         super().__init__()
#         self.attention = attention_module
        
#     def forward(self, x):
#         return self.attention(x)


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
        for layer in self.layers:
            x = layer(x, reverse=False)
        return x

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        return x


if __name__ == "__main__":
    net = InvertibleColorTransform()
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