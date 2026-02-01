import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Dict
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import Bottleneck, C2f

class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MDJA(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_threshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.GMSA = GMSA(op_channel,
                       group_num=group_num,
                       gate_threshold=gate_threshold)
        self.AGC = AGC(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
        # self.conv = nn.Conv2d(op_channel * 2, op_channel, kernel_size=1)

    def forward(self, x):
        x = self.GMSA(x)
        x = self.AGC(x)

        # x = self.conv(torch.cat([x1, x2], dim=1))
        return x

class ConvBNReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(ConvBNReLUBlock, self).__init__()
        self.size = size
        self.conv_bn_silu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=size, padding=(size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv_bn_silu(x)
        return out

class TQ(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TQ, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_a = x
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x)
        out = x_a + x_1 + x_2 + x_3 + x_4
        return out

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class GMSA(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()
        self.tq = TQ(oup_channels, oup_channels)

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        x = x * reweigts
        out = self.tq(x)

        return out

class AGC(nn.Module):
    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Sequential(
            # GCT(up_channel),
            nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        )
        self.squeeze2 = nn.Sequential(
            # GCT(low_channel),
            nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        )
        # up
        self.GWC = nn.Sequential(
            # GCT(up_channel // squeeze_radio),
            nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                      padding=group_kernel_size // 2, groups=group_size)
        )
        self.PWC1 = nn.Sequential(
            # GCT(up_channel // squeeze_radio),
            nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        )
        # low
        self.PWC2 = nn.Sequential(
            # GCT(low_channel // squeeze_radio),
            nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                      bias=False)
        )
        self.advavg = nn.AdaptiveAvgPool2d(1)
        # self.channelattention = ChannelAttention(op_channel, op_channel // 4)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        # out = torch.cat([Y1, Y2], dim=1)
        # out = F.softmax(self.advavg(out), dim=1) * out
        # out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        out1 = F.softmax(self.advavg(Y1), dim=1) * x
        out2 = F.softmax(self.advavg(Y2), dim=1) * x
        out = out1 + out2 + x
        #        out1 = self.channelattention(Y1) * x
        #  out2 = self.channelattention(Y2) * x
        return out