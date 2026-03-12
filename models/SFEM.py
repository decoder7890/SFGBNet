import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class CWCB(nn.Module):
    def __init__(self, in_channels):
        super(CWCB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # Interleaved concatenation
        b, c, h, w = x1.size()
        x = torch.zeros(b, 2 * c, h, w, device=x1.device)
        x[:, 0::2, :, :] = x1
        x[:, 1::2, :, :] = x2

        # Depthwise convolution
        out = self.conv(x)
        return out


class DWT(nn.Module):
    """离散小波变换封装。"""

    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
        self.DWT = DWTForward(J=1, wave="haar")

    def forward(self, x):
        # 使用 DWTForward 计算离散小波变换
        return self.DWT(x)


class IWT(nn.Module):
    """逆离散小波变换封装。"""

    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
        self.IDWT = DWTInverse(wave="haar")

    def forward(self, x):
        # 使用 DWTInverse 计算逆离散小波变换
        return self.IDWT(x)


class RHDWT(nn.Module):
    def __init__(self, in_channels, n: int = 1):
        super(RHDWT, self).__init__()
        # 卷积操作, 用于后续的反向传递/残差
        self.identety = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * n,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        # DWT (离散小波变换)
        self.DWT = DWTForward(J=1, wave="haar")
        # 编码部分
        self.dconv_encode = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * n, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        # IDWT (离散小波逆变换) 预留
        self.IDWT = DWTInverse(wave="haar")

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        a = DMT1_yh[0]  # (B, C, 3, H', W')
        list_tensor.append(DMT1_yl)
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)

    def forward(self, x):
        input = x
        # 获取 DWT 输出 (低频和高频部分)
        DMT1_yl, DMT1_yh = self.DWT(x)
        # 变换操作, 将高频部分合并到一起
        DMT = self._transformer(DMT1_yl, DMT1_yh)
        # 编码操作
        x = self.dconv_encode(DMT)
        # 身份卷积操作, 残差连接
        res = self.identety(input)
        # 输出加上残差
        out = torch.add(x, res)
        return out