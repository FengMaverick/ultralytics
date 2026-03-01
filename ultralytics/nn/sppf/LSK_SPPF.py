import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

# -----------------------------------------
# LSK-SPPF (替代原 SPPF)
# -----------------------------------------
class LSK_SPPF(nn.Module):
    """
    LSK-based SPPF
    Large Selective Kernel with lightweight design
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()

        c_ = c1 // 2  # hidden channels

        # 降维
        self.cv1 = Conv(c1, c_, 1, 1)

        # 多尺度大核 depthwise 卷积
        self.dw5  = nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False)
        self.dw9  = nn.Conv2d(c_, c_, 9, 1, 4, groups=c_, bias=False)
        self.dw13 = nn.Conv2d(c_, c_, 13, 1, 6, groups=c_, bias=False)

        self.bn = nn.BatchNorm2d(c_)

        # 选择权重生成（轻量）
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_, c_ // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_ // 4, 3, 1, bias=True)
        )

        # 输出融合
        self.cv2 = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):

        x = self.cv1(x)

        # 三种感受野
        x1 = self.bn(self.dw5(x))
        x2 = self.bn(self.dw9(x))
        x3 = self.bn(self.dw13(x))

        # 计算选择权重
        attn = self.attn(x)
        attn = F.softmax(attn, dim=1)

        # 分解权重
        w1 = attn[:, 0:1]
        w2 = attn[:, 1:2]
        w3 = attn[:, 2:3]

        # 加权融合
        out = w1 * x1 + w2 * x2 + w3 * x3

        # 拼接原始特征
        out = torch.cat([x, out], dim=1)

        return self.cv2(out)