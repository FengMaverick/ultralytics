import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import CondConv2d
from ultralytics.nn.modules.conv import autopad
class DynamicConv_Single(nn.Module):
    """ Dynamic Conv layer
    """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)
        
    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x

class DynamicConv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, num_experts=4):
        super().__init__()
        self.conv = nn.Sequential(
            DynamicConv_Single(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), dilation=d, groups=g, num_experts=num_experts),
            nn.BatchNorm2d(c2),
            self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)