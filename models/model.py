import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DTAM import DTAMFusion_Variant
from models.FGFM import Diff
from models.SFEM import CWCB, RHDWT
from models.PU_S import PU, PS
from models.resnet import resnet18


class HSSNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(HSSNet, self).__init__()
        self.backbone = resnet18(pretrained=True)  # 使用resnet18

        self.mid_d = 64
        # DTAMFusion-based feature pair enhancement modules
        self.DTAM5 = DTAMFusion_Variant(512)  # 512 8 8
        self.DTAM4 = DTAMFusion_Variant(256)  # 256 16 16
        self.DTAM3 = DTAMFusion_Variant(128)  # 128 32 32
        self.DTAM2 = DTAMFusion_Variant(64)   # 64 64 64

        self.CWCB5 = CWCB(512)
        self.CWCB4 = CWCB(256)
        self.CWCB3 = CWCB(128)
        self.CWCB2 = CWCB(64)

        # SFEM modules to process CWCB outputs
        self.SFEM5 = RHDWT(512, n=1)
        self.SFEM4 = RHDWT(256, n=1)
        self.SFEM3 = RHDWT(128, n=1)
        self.SFEM2 = RHDWT(64, n=1)

        # FGFM
        self.FGFM5 = Diff(512)
        self.FGFM4 = Diff(256)
        self.FGFM3 = Diff(128)
        self.FGFM2 = Diff(64)

        # PU/S 多尺度下采样/上采样融合
        # K4(64, 1/4) + K3(128, 1/8) -> Z1(128, 1/8)
        # Z1(128, 1/8) + K2(256, 1/16) -> Z2(256, 1/16)
        # Z2(256, 1/16) + K1(512, 1/32) -> Z3(512, 1/32)
        self.PU1 = PU(64, 128, 128, block_size=2, k_size=3)
        self.PU2 = PU(128, 256, 256, block_size=2, k_size=3)
        self.PU3 = PU(256, 512, 512, block_size=2, k_size=3)

        # 自底向上上采样融合
        # Z3(512, 1/32) + Z2(256, 1/16) -> A1(256, 1/16)
        # A1(256, 1/16) + Z1(128, 1/8) -> A2(128, 1/8)
        # A2(128, 1/8) + K4(64, 1/4) -> A3(64, 1/4)
        self.PS1 = PS(512, 256, 256, block_size=2, k_size=3)
        self.PS2 = PS(256, 128, 128, block_size=2, k_size=3)
        self.PS3 = PS(128, 64, 64, block_size=2, k_size=3)

        # 解码器部分（使用 Z3, A1, A2, A3）
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.channels_cut5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.channels_cut4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.channels_cut3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        
        # 改进：使用更强的最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        # DTAMFusion_Variant，得到两路增强特征
        Y1_5, Y2_5 = self.DTAM5(x1_5, x2_5)  # 1/32
        Y1_4, Y2_4 = self.DTAM4(x1_4, x2_4)  # 1/16
        Y1_3, Y2_3 = self.DTAM3(x1_3, x2_3)  # 1/8
        Y1_2, Y2_2 = self.DTAM2(x1_2, x2_2)  # 1/4

        C1 = self.CWCB5(x1_5, x2_5)
        C2 = self.CWCB4(x1_4, x2_4)
        C3 = self.CWCB3(x1_3, x2_3)
        C4 = self.CWCB2(x1_2, x2_2)

        # CWCB 输出进入 SFEM，得到第三路特征
        R1 = self.SFEM5(C1)
        R2 = self.SFEM4(C2)
        R3 = self.SFEM3(C3)
        R4 = self.SFEM2(C4)

        # 三路特征在 FGFM 中进行融合
        K1 = self.FGFM5(R1, Y1_5, Y2_5)
        K2 = self.FGFM4(R2, Y1_4, Y2_4)
        K3 = self.FGFM3(R3, Y1_3, Y2_3)
        K4 = self.FGFM2(R4, Y1_2, Y2_2)

        # PU/S 多尺度融合：K4 & K3 -> Z1, Z1 & K2 -> Z2, Z2 & K1 -> Z3
        Z1 = self.PU1(K4, K3)   # 通道: 128, 尺度: 1/8
        Z2 = self.PU2(Z1, K2)   # 通道: 256, 尺度: 1/16
        Z3 = self.PU3(Z2, K1)   # 通道: 512, 尺度: 1/32

        # 自底向上 PU/S：Z3 & Z2 -> A1, A1 & Z1 -> A2, A2 & K4 -> A3
        A1 = self.PS1(Z3, Z2)   # 通道: 256, 尺度: 1/16
        A2 = self.PS2(A1, Z1)   # 通道: 128, 尺度: 1/8
        A3 = self.PS3(A2, K4)   # 通道: 64,  尺度: 1/4

        # 解码器部分：使用 Z3, A1, A2, A3 代替原来的 K1~K4
        D1 = torch.cat((self.upconv5(Z3), A1), dim=1)
        D1 = self.channels_cut5(D1)
        D2 = torch.cat((self.upconv4(D1), A2), dim=1)
        D2 = self.channels_cut4(D2)
        D3 = torch.cat((self.upconv3(D2), A3), dim=1)
        D3 = self.channels_cut3(D3)
        
        # 改进：使用双线性插值 + 更强的输出层
        mask = F.interpolate(D3, x1.size()[2:], mode='bilinear', align_corners=True)
        output = self.final_conv(mask)
        output = torch.sigmoid(output)
        return output


if __name__ == '__main__':
    x1 = torch.randn((4, 3, 256, 256)).cuda()
    x2 = torch.randn((4, 3, 256, 256)).cuda()
    model = HSSNet(3, 1).cuda()
    out = model(x1, x2)
    print(out.shape)
