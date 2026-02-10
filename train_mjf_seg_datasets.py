from ultralytics import YOLO

def main_pt():
    # 1. 加载模型
    model = YOLO("yolo11s-seg.pt")

    # 2. 开始训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/nnew_mjf_seg_datasets_yolo_V3/dataset.yaml",
        epochs=500,
        imgsz=640,
        batch=32,
        device=0,
        name="train-nnmsdyV3-yolo11s_seg_e500", # 实验名称, nnmsdyV3是缩写，ls500表示label_smoothing和500epochs（默认300），e500是500epochs
        workers=8,   # 数据加载线程数
        exist_ok=True, # 允许覆盖现有的 project/name 目录

        # label_smoothing=0.1, # 跑出来效果不行

        optimizer='SGD', # 训练优化器的选择。选项包括 SGD, Adam, AdamW, NAdam, RAdam, RMSProp 等等，或者 auto 用于基于模型配置自动选择。影响收敛速度和稳定性。
        lr0=0.01,        # 初始学习率（即 SGD=1E-2, Adam=1E-3)。调整此值对于优化过程至关重要，它会影响模型权重更新的速度。
        momentum=0.937,  # SGD 的动量因子或 Adam 优化器 的 beta1，影响当前更新中过去梯度的整合。
        cos_lr=True,     # 开启余弦退火

        # === 数据增强===
        copy_paste=0.1,  # 跨图像复制和粘贴对象以增加对象实例。
        mixup=0.1,       # 混合两个图像及其标签，创建一个合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。
        degrees=10.0,    # 在指定的角度范围内随机旋转图像，提高模型识别各种方向物体的能力。
        cutmix=0.1       # 组合两张图像的部分区域，创建局部混合，同时保持清晰的区域。通过创建遮挡场景来增强模型的鲁棒性。
    )

def main_yaml():
    # step 1: 实例化模型
    # 注意：这里必须填你的【YAML文件】，而不是 .pt 文件
    # 这告诉 YOLO：先按这个 YAML 搭积木，把网络层建好（此时参数是随机的）
    model = YOLO("/home/aa205/MJF/ultralytics/ultralytics/cfg/models/11/yolo11s-seg-C3k2_DCNv4_V2.yaml")

    # step 2: 加载预训练权重 (迁移学习的关键)
    # 这告诉 YOLO：去官方权重里找名字一样的层，把参数复制过来
    # 控制台会显示 "Transferred X/Y items..."
    model.load("yolo11s-seg.pt")

    # step 3: 开始训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/nnew_mjf_seg_datasets_yolo_V3/dataset.yaml",
        epochs=500,
        # patience=50,       #【新增】早停机制：50轮不涨就停
        imgsz=640,
        batch=32,
        device=0,
        name="train-nnmsdyV3-yolo11s_seg_C3k2_DCNv4_V2_e500",  # 实验名称
        workers=8,  # 数据加载线程数
        exist_ok=True,  # 允许覆盖现有的 project/name 目录
        # deterministic=False, # 为True，强制使用确定性算法

        # label_smoothing=0.1,

        optimizer='SGD', # 训练优化器的选择。选项包括 SGD, Adam, AdamW, NAdam, RAdam, RMSProp 等等，或者 auto 用于基于模型配置自动选择。影响收敛速度和稳定性。
        lr0=0.01,        # 初始学习率（即 SGD=1E-2, Adam=1E-3)。调整此值对于优化过程至关重要，它会影响模型权重更新的速度。
        momentum=0.937,  # SGD 的动量因子或 Adam 优化器 的 beta1，影响当前更新中过去梯度的整合。
        cos_lr=True,     # 开启余弦退火

        # === 数据增强 ===
        copy_paste=0.1,  # 跨图像复制和粘贴对象以增加对象实例。
        mixup=0.1,       # 混合两个图像及其标签，创建一个合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。
        degrees=10.0,    # 在指定的角度范围内随机旋转图像，提高模型识别各种方向物体的能力。
        cutmix=0.1       # 组合两张图像的部分区域，创建局部混合，同时保持清晰的区域。通过创建遮挡场景来增强模型的鲁棒性。
    )

if __name__ == "__main__":
    main_pt()
    # main_yaml()