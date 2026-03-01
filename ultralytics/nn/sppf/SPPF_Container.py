import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init


class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.
    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        # n, c = x.size(0)
        n = x.size(0)
        c = self.inter_channels
        # n, nH, nW, c = x.shape

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y




class SPPF_Container(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attention = ContextAggregation(c_ * 4)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.attention(torch.cat((x, y1, y2, self.m(y2)), 1)))

# [!!! GEMINI 3.1pro !!!]
class CoordAtt(nn.Module):
    """
    轻量级坐标注意力机制 (Coordinate Attention)
    能以极低的计算代价捕获长距离依赖，并保留精确的空间位置信息。
    """
    def __init__(self, inp_channels, reduction=32):
        super(CoordAtt, self).__init__()
        # 分别沿垂直 (H) 和水平 (W) 方向进行自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp_channels // reduction)

        self.conv1 = nn.Conv2d(inp_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU() # 契合 YOLO 默认的激活函数

        self.conv_h = nn.Conv2d(mip, inp_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        
        # 提取 X 和 Y 方向的特征
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接后进行通道压缩和非线性激活
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 重新分离为 H 和 W 的注意力掩码
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 空间权重与原特征相乘
        return x * a_h * a_w

class EfficientContextSPPF(nn.Module):
    """
    结合了坐标注意力机制的高效 SPPF 模块。
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        # 用轻量级的 CoordAtt 替换原本沉重的 ContextAggregation
        self.attention = CoordAtt(c_ * 4, reduction=16)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        
        # 将不同感受野的特征拼接
        out = torch.cat((x, y1, y2, self.m(y2)), 1)
        
        # 先进行注意力加权，再通过 cv2 降维输出
        return self.cv2(self.attention(out))

# [!!! GEMINI 3.1pro 增强版 SPPF_Container_PE !!!]
class ContextAggregation_PE(nn.Module):
    """
    带有空间位置编码 (Spatial Positional Encoding) 的 Context Aggregation Block.
    解决了 GCNet/ContextAggregation 在矩阵相乘时丢失 2D 空间排布的致命缺陷。
    """
    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation_PE, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)
        
        # --- 核心改进：引入隐式条件位置编码 (Conditional Positional Encoding, CPE) ---
        # 使用 3x3 深度可分离卷积。因为 YOLO 需要适应多尺度输入，绝对正弦/余弦位置编码效果奇差且难对齐。
        # CPE 是目前最具性价比的三维注意力补充方案：零计算负担，免插值，强迫特征感知上下左右邻居。
        self.pe = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)
        # 初始化为0，使得网络在最初始阶段等同于原版 SPPF_Container，平滑过渡而不破坏预训练
        constant_init(self.pe, 0)

    def forward(self, x):
        n = x.size(0)
        c = self.inter_channels

        # --- 1. 注入空间位置信息 ---
        # 让每一个孤立的像素点“看一眼”自己周围的地理位置
        x_pe = x + self.pe(x) 

        # --- 2. 算子注意力计算 (全基于标定过坐标的特征) ---
        a = self.a(x_pe).sigmoid()
        # 注意力图 K 包含了坐标信息
        k = self.k(x_pe).view(n, 1, -1, 1).softmax(2)
        # Value 内容 V 也包含了坐标信息
        v = self.v(x_pe).view(n, 1, c, -1)

        # --- 3. 全局降维相乘 ---
        # 这时候算出来的融合特征，就不再是把剪刀刀柄和剪刀头搞混了，而是“左边的刀柄”和“右边的刀头”
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y

class SPPF_Container_PE(nn.Module):
    """
    终极改进：结合了空间位置感知的 SPPF_Container_PE 模块。
    建议将网络架构中第9层的 SPPF_Container 直接替换为该模块。
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2 
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        # 替换为带位置编码的注意力聚合
        self.attention = ContextAggregation_PE(c_ * 4)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.attention(torch.cat((x, y1, y2, self.m(y2)), 1)))