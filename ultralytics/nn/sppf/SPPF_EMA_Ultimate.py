import torch
import torch.nn as nn
import warnings
from ultralytics.nn.modules import Conv
from ultralytics.nn.attention.attention import EMA

class SPPF_EMA_Ultimate(nn.Module):
    """
    终极 SPPF 版本：结合了局部多尺度池化、全局池化上下文，
    并使用 EMA (Efficient Multi-Scale Attention) 处理庞大维度。
    EMA 天生适合处理多尺度重叠特征，并且能够完美保持空间极化方向。
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2 
        # 初始降维
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # 拼接 6 层特征 (原始, 3个局部Max, 1个全局Max, 1个全局Avg)，所以输入通道数为 c_ * 6
        self.cv2 = Conv(c_ * 6, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        # 全局池化算子
        self.am = nn.AdaptiveMaxPool2d(1)
        self.aa = nn.AdaptiveAvgPool2d(1)
        
        # 核心替换：使用 EMA (Efficient Multi-Scale Attention)
        # Factor 是分组数，c_ * 6 通常能被 8 整除 (YOLO通道多为16/32的倍数)
        # EMA 能直接把这 6 份特征按照跨组通道和空间维度去粗取精，且不开销太多算力。
        self.attention = EMA(c_ * 6, factor=8)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            # 1. 局部层次感受野
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2) 
            
            # 2. 全局图像级感受野 
            g_max = self.am(x).expand_as(x)
            g_avg = self.aa(x).expand_as(x)
            
            # 3. 暴力拼接所有维度的特征 (此时通道激增)
            concat_out = torch.cat((x, y1, y2, y3, g_max, g_avg), 1)
            
            # 4. EMA 注意力引擎：高效过滤冗余和跨维度感知
            attn_out = self.attention(concat_out)
            
            # 5. 回收降维至目标通道数
            return self.cv2(attn_out)
