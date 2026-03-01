import torch
import torch.nn as nn

class LSKBlock(nn.Module):
    """
    LSK (Large Selective Kernel) Attention Block.
    Reference: LSKNet: Large Selective Kernel Network for Remote Sensing Object Detection (ICCV 2023)
    Designed to capture long-range spatial contexts with large kernels and dynamic selection.
    """
    def __init__(self, dim):
        super().__init__()
        # Branch 1: Large kernel (5x5, implicit receptive field approx 5)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # Branch 2: Even larger effective kernel (7x7 dilation 3 -> RF 19x19)
        # This acts as the "Global Context Radar"
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        
        # Channel mixing layers
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        
        # Squeeze-and-Excitation style fusion
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv3 = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        
        # Spatial Attention Map
        sig = self.conv_squeeze(agg).sigmoid()
        
        # Dynamic Selection
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv3(attn)
        return x * attn
