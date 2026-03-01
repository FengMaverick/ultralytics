import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNFusion(nn.Module):
    """
    Stable BiFPN weighted feature fusion.
    Replace Concat in YOLO Neck.
    """

    def __init__(self, c1, c2, out_channels):
        """
        c1: channels of input 1
        c2: channels of input 2
        out_channels: output channels after fusion
        """
        super().__init__()

        # 通道对齐
        self.align1 = nn.Conv2d(c1, out_channels, 1, 1, bias=False)
        self.align2 = nn.Conv2d(c2, out_channels, 1, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 可学习权重（2个输入）
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

        # 融合后 refine
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        x1, x2 = x

        # 通道对齐
        x1 = self.bn1(self.align1(x1))
        x2 = self.bn2(self.align2(x2))

        # 非负权重
        w = F.relu(self.w)

        # 归一化
        weight = w / (torch.sum(w) + self.epsilon)

        # 加权求和（不是concat）
        out = weight[0] * x1 + weight[1] * x2

        # refine
        out = self.out_conv(out)

        return out

