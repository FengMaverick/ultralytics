import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv,autopad
from ultralytics.nn.modules.block import Bottleneck, C2f,C3k


try:
    from DCNv4.modules.dcnv4 import DCNv4
except ImportError as e:
    pass


class DCNV4_YOLO(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        
        dcn_group = g
        # DCNv4 requirement: channels // group must be divisible by 16
        # Optimization: Try to use group=4 if possible (needs channels >= 64 and divisible by 64)
        if g == 1 and ouc % (4 * 16) == 0:
            dcn_group = 4

        self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=dcn_group, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(N, -1, C)
        x = self.dcnv4(x, (H, W))
        x = x.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.act(self.bn(x))
        return x

class Bottleneck_DCNV4(Bottleneck):
    """Standard bottleneck with DCNV4."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # c2 must be output of DCNv4, k[1] is the DCN kernel size
        self.cv2 = DCNV4_YOLO(c_, c2, k=k[1], g=g)

class C3k_DCNv4(C3k):
    """C3k with DCNv4."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DCNV4(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DCNv4(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, k=3):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_DCNv4(self.c, self.c, 2, shortcut, g, k=k) if c3k else Bottleneck_DCNV4(self.c, self.c, shortcut, g, k=(k, k)) for _ in range(n)
        )