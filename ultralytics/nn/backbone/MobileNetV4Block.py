import torch
import torch.nn as nn
from typing import Optional

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2, C3k

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
    ) -> int:
    """
    Ensure that all layers have channels that are divisible by 8.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

class MobileNetV4Block(nn.Module):
    """
    MobileNetV4 Universal Inverted Bottleneck (UIB) Block.
    Adapted for YOLOv11: Uses SiLU activation (via Ultralytics Conv) instead of ReLU6.
    """
    def __init__(self, 
            c1, 
            c2, 
            start_dw_kernel_size=3,
            middle_dw_kernel_size=5,
            middle_dw_downsample=True,
            stride=1,
            expand_ratio=2 # Adjusted default to be closer to YOLO bottleneck
        ):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.stride = stride
        
        # 1. Starting Depthwise Conv (Optional)
        if self.start_dw_kernel_size > 0:            
            # If downsampling happens in middle, start DW keeps stride 1. If not, it takes the stride.
            # But usually UIB does downsampling in the middle or end.
            # For C3k2 integration, stride is usually 1 (bottleneck).
            stride_ = stride if not middle_dw_downsample else 1
            self.start_dw_conv = Conv(c1, c1, k=start_dw_kernel_size, s=stride_, g=c1, act=True) # Use default SiLU
        else:
            self.start_dw_conv = nn.Identity()

        # 2. Pointwise Expansion Conv
        hidden_dim = make_divisible(c1 * expand_ratio, 8)
        self.expand_conv = Conv(c1, hidden_dim, k=1, s=1, act=True) # Use default SiLU

        # 3. Middle Depthwise Conv (Optional)
        if self.middle_dw_kernel_size > 0:
            stride_ = stride if middle_dw_downsample else 1
            self.middle_dw_conv = Conv(hidden_dim, hidden_dim, k=middle_dw_kernel_size, s=stride_, g=hidden_dim, act=True) # Use default SiLU
        else:
             self.middle_dw_conv = nn.Identity()

        # 4. Pointwise Projection Conv (Linear)
        # Note: MobileNet uses linear here. We use act=False (Linear)
        self.project_conv = Conv(hidden_dim, c2, k=1, s=1, act=False) 

    def forward(self, x):
        if self.start_dw_kernel_size > 0:
            x = self.start_dw_conv(x)
        x = self.expand_conv(x)
        if self.middle_dw_kernel_size > 0:
            x = self.middle_dw_conv(x)
        x = self.project_conv(x)
        return x

class C3k_MobileNetV4(C3k):
    """
    C3k Block with MobileNetV4Block as the bottleneck.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        # Initialize C3k but override its m (bottleneck module) sequence
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        # Replace the standard C3k/Bottleneck with MobileNetV4Block
        # We adapt arguments: MobileNetV4Block(c1, c2)
        # Use a slightly larger expand ratio for better feature extraction in the bottleneck
        self.m = nn.Sequential(*(MobileNetV4Block(c_, c_, start_dw_kernel_size=3, middle_dw_kernel_size=5, expand_ratio=2) 
                                 for _ in range(n)))

class C3k2_MobileNetV4(C3k2):
    """
    C3k2 Block using C3k_MobileNetV4.
    This replaces the standard C3k inside C3k2 with our MobileNetV4-enhanced version.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        
        # Calculate hidden channels
        c_ = int(c2 * e)
        
        # Override self.m with standard nn.Sequential (CRITICAL FIX from user code)
        if c3k:
             self.m = nn.ModuleList(C3k_MobileNetV4(c_, c_, 2, shortcut, g) for _ in range(n))
        else:
             self.m = nn.ModuleList(MobileNetV4Block(c_, c_, start_dw_kernel_size=3, middle_dw_kernel_size=5, expand_ratio=2) 
                                      for _ in range(n))
