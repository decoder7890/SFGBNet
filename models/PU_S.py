import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        y = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
        # 只需要升维到 4 维与 x 对齐即可：[B, C, 1, 1]
        y = torch.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # [B, C, 1, 1]
        return x * y.expand_as(x)


class Space2Depth(nn.Module):
    def __init__(self, block_size=2):
        super(Space2Depth, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.size()
        assert H % self.block_size == 0 and W % self.block_size == 0
        x = x.view(B, C, H // self.block_size, self.block_size, W // self.block_size, self.block_size)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * self.block_size * self.block_size, H // self.block_size, W // self.block_size)
        return x


class Depth2Space(nn.Module):
    def __init__(self, block_size=2):
        super(Depth2Space, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C_mul, H, W = x.size()
        C = C_mul // (self.block_size ** 2)
        assert C_mul % (self.block_size ** 2) == 0
        x = x.view(B, C, self.block_size, self.block_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C, H * self.block_size, W * self.block_size)
        return x


class PU(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, block_size=2, k_size=3):
        super(PU, self).__init__()
        self.space2depth = Space2Depth(block_size)
        self.conv1 = nn.Conv2d(
            in_channels1 * block_size * block_size,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=block_size * block_size,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.PReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU()

        self.eca = ECA(out_channels, k_size)

    def forward(self, x1, x2):
        # x1: 高分辨率主分支 [B, C1, H, W]
        # x2: 辅助分支 [B, out_channels, H/2, W/2]
        x1_down = self.space2depth(x1)  # [B, C1*4, H/2, W/2]
        x1_feat = self.act1(self.bn1(self.conv1(x1_down)))  # [B, out_channels, H/2, W/2]
        y = x1_feat + x2  # 融合
        x2_feat = self.act2(self.bn2(self.conv2(y)))  # 再处理
        out = self.eca(x2_feat + y)  # 残差融合+ECA
        return out


class PS(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, block_size=2, k_size=3):
        super(PS, self).__init__()
        self.depth2space = Depth2Space(block_size)
        self.conv1 = nn.Conv2d(in_channels1 // (block_size * block_size), out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.PReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU()

        self.eca = ECA(out_channels, k_size)

    def forward(self, x1, x2):
        # x1: 低分辨率主分支 [B, C1, H/2, W/2]
        # x2: 辅助分支 [B, out_channels, H, W]
        x1_up = self.depth2space(x1)  # [B, C1//4, H, W]
        x1_feat = self.act1(self.bn1(self.conv1(x1_up)))  # [B, out_channels, H, W]
        y = x1_feat + x2  # 融合
        x2_feat = self.act2(self.bn2(self.conv2(y)))  # 再处理
        out = self.eca(x2_feat + y)  # 残差融合+ECA
        return out


