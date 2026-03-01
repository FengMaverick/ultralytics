import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2 
try:
    from DCNv4.modules.dcnv4 import DCNv4
except ImportError as e:
    pass

class DCNv4_YOLO(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        if ouc % 4 == 0 and (ouc // 4) % 16 == 0:
            dcn_group = 4
        else:
            dcn_group = 1 
        self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=dcn_group, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        N, C, H, W = x.shape
        x_permuted = x.permute(0, 2, 3, 1).contiguous().view(N, -1, C)
        x_out = self.dcnv4(x_permuted, (H, W))
        x_out = x_out.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_out = self.act(self.bn(x_out))
        return x_out

class Bottleneck_DCNv4(Bottleneck):
    """Standard bottleneck with DCNv4."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv4_YOLO(c_, c2, k[1])

class C3k_DCNv4(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNv4(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DCNv4(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DCNv4(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DCNv4(self.c, self.c, shortcut, g) for _ in range(n))