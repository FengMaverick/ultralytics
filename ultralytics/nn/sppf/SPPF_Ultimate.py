import torch
import torch.nn as nn
import warnings
from ultralytics.nn.modules import Conv

# 从你的已有文件中导入带有位置信息的 Attention
from .SPPF_Container import ContextAggregation_PE 

class SPPF_Ultimate(nn.Module):
    """
    结合了多尺度局部池化、全局池化上下文，以及带有位置编码的注意力机制的最终 SPPF 版本。
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
        
        # 核心：处理 6 倍庞大拼接特征的注意力引擎
        # 强烈建议使用含有 PE (位置编码) 的版本，保障分割任务的像素掩码不移位
        self.attention = ContextAggregation_PE(c_ * 6)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            # 1. 局部层次感受野
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)  # 原代码用 self.m(y2)，为保持对应关系显式拉出 y3
            
            # 2. 全局图像级感受野 (平铺延展对齐到原始 HxW 大小)
            g_max = self.am(x).expand_as(x)
            g_avg = self.aa(x).expand_as(x)
            
            # 3. 暴力拼接所有特征 (此时通道数激增至 c_ * 6)
            concat_out = torch.cat((x, y1, y2, y3, g_max, g_avg), 1)
            
            # 4. 让 Attention 判断这厚重的特征墙中，哪些通道有效，哪些是冗余噪声
            attn_out = self.attention(concat_out)
            
            # 5. 最后再由 1x1 卷积平滑降压到指定的 c2 输出维度
            return self.cv2(attn_out)
