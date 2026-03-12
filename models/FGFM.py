import torch
import torch.nn as nn
import torch.nn.functional as F

class Diff(nn.Module):
    def __init__(self, in_dim):
        super(Diff, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, F3, F1, F2):
        """
        F3: feature from SFEM branch (may have different spatial size)
        F1: feature from DTAM branch
        F2: feature from DTAM branch

        在原始实现中，假设 F3、F1、F2 的空间尺寸完全一致。但在当前网络中，
        SFEM 分支会对特征做下采样，导致 F3 的空间尺寸比 F1/F2 更小，
        从而在 view + bmm 时出现 size mismatch。

        这里显式地将 F3 和 F2 的空间分辨率对齐到 F1，以保证注意力计算的
        形状一致，避免 RuntimeError。
        """
        m_batchsize, C, height, width = F1.size()

        # 对齐空间尺寸到 F1
        if F3.size()[2:] != (height, width):
            F3 = F.interpolate(F3, size=(height, width), mode="bilinear", align_corners=False)
        if F2.size()[2:] != (height, width):
            F2 = F.interpolate(F2, size=(height, width), mode="bilinear", align_corners=False)

        # Use F3 as Q
        query = self.query_conv(F3)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)

        # Use F1 as K
        key = self.key_conv(F1)
        proj_key = key.view(m_batchsize, -1, width * height)

        # Use F2 as V
        value = self.value_conv(F2)
        proj_value = value.view(m_batchsize, -1, width * height)

        # Scaled dot-product attention
        energy = torch.bmm(proj_query, proj_key) / (C // 8) ** 0.5
        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + F3
        return out
