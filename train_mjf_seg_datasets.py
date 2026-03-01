from ultralytics import YOLO

def main_pt():
    # 1. 加载模型
    model = YOLO("yolo11s-seg.pt")

    # 2. 开始训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/new_mjf_seg_datasets_yolo_V3/dataset.yaml",
        epochs=300,
        imgsz=640,
        batch=32,
        device=0,
        name="train-nmsdyV3-yolo11s_seg",
        workers=8,   # 数据加载线程数
        exist_ok=True, # 允许覆盖现有的 project/name 目录

        # optimizer='SGD', # 训练优化器的选择。选项包括 SGD, Adam, AdamW, NAdam, RAdam, RMSProp 等等，或者 auto 用于基于模型配置自动选择。影响收敛速度和稳定性。
        # lr0=0.01,        # 初始学习率（即 SGD=1E-2, Adam=1E-3)。调整此值对于优化过程至关重要，它会影响模型权重更新的速度。

        # === 数据增强===
        copy_paste=0.3,  # 跨图像复制和粘贴对象以增加对象实例。
        mixup=0.1,       # 混合两个图像及其标签，创建一个合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。
        cutmix=0.1     # 组合两张图像的部分区域，创建局部混合，同时保持清晰的区域。通过创建遮挡场景来增强模型的鲁棒性。
    )

def main_yaml():
    # step 1: 实例化模型
    # 注意：这里必须填你的【YAML文件】，而不是 .pt 文件
    # 这告诉 YOLO：先按这个 YAML 搭积木，把网络层建好（此时参数是随机的）
    model = YOLO("/home/aa205/MJF/ultralytics/ultralytics/cfg/models/11/yolo11s-seg-EUCB.yaml")

    # step 2: 加载预训练权重 (迁移学习的关键)
    # 这告诉 YOLO：去官方权重里找名字一样的层，把参数复制过来
    # 控制台会显示 "Transferred X/Y items..."
    model.load("yolo11s-seg.pt")

    # 打印 Segment 模块结构
    # for name, m in model.model.named_modules():
    #     if type(m).__name__ == "Segment":
    #         print(f"\n{'='*60}")
    #         print(f"Segment 模块结构 (name={name}):")
    #         print(f"{'='*60}")
    #         print(m)
    #         print(f"{'='*60}\n")

    # step 3: 开始训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V3/dataset.yaml",
        epochs=300,
        # patience=50,       #【新增】早停机制：50轮不涨就停
        imgsz=640,
        batch=32,
        device=0,
        name="train-msdyV3-yolo11s_seg_EUCB",  # 实验名称
        workers=8,  # 数据加载线程数
        exist_ok=True,  # 允许覆盖现有的 project/name 目录
        # deterministic=False, # 为True，强制使用确定性算法
        # amp = False, # TRUE : 启用自动混合精度（AMP）训练，减少内存使用，并可能在对准确性影响最小的情况下加快训练速度。

        # optimizer='SGD', # 训练优化器的选择。选项包括 SGD, Adam, AdamW, NAdam, RAdam, RMSProp 等等，或者 auto 用于基于模型配置自动选择。影响收敛速度和稳定性。
        # lr0=0.01,        # 初始学习率（即 SGD=1E-2, Adam=1E-3)。调整此值对于优化过程至关重要，它会影响模型权重更新的速度。

        # === 数据增强 ===
        copy_paste=0.3,  # 跨图像复制和粘贴对象以增加对象实例。
        mixup=0.1,       # 混合两个图像及其标签，创建一个合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。
        cutmix=0.1       # 组合两张图像的部分区域，创建局部混合，同时保持清晰的区域。通过创建遮挡场景来增强模型的鲁棒性。
    )

def main_yaml1():
    # step 1: 实例化模型
    # 注意：这里必须填你的【YAML文件】，而不是 .pt 文件
    # 这告诉 YOLO：先按这个 YAML 搭积木，把网络层建好（此时参数是随机的）
    model = YOLO("/home/aa205/MJF/ultralytics/ultralytics/cfg/models/11/yolo11s-seg-C3k2_DCNv4-SPPF_Container-EUCB.yaml")

    # step 2: 加载预训练权重 (迁移学习的关键)
    # 这告诉 YOLO：去官方权重里找名字一样的层，把参数复制过来
    # 控制台会显示 "Transferred X/Y items..."
    model.load("yolo11s-seg.pt")

    # 打印 Segment 模块结构
    # for name, m in model.model.named_modules():
    #     if type(m).__name__ == "Segment":
    #         print(f"\n{'='*60}")
    #         print(f"Segment 模块结构 (name={name}):")
    #         print(f"{'='*60}")
    #         print(m)
    #         print(f"{'='*60}\n")

    # step 3: 开始训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V3/dataset.yaml",
        epochs=500,
        # patience=50,       #【新增】早停机制：50轮不涨就停
        imgsz=640,
        batch=32,
        device=0,
        name="train-msdyV3-yolo11s_seg_C3k2_DCNv4_SPPF_Container_EUCB",  # 实验名称
        workers=8,  # 数据加载线程数
        exist_ok=True,  # 允许覆盖现有的 project/name 目录

        weight_decay=0.001, # L2 正则化项，惩罚大权重以防止过拟合。
        dropout=0.1, # 分类任务中用于正则化的 Dropout 率，通过在训练期间随机省略单元来防止过拟合。

        # === 数据增强 ===
        copy_paste=0.3,   # 跨图像复制和粘贴对象以增加对象实例。
        mixup=0.1,        # 混合两个图像及其标签，创建一个合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。
        cutmix=0.1,       # 组合两张图像的部分区域，创建局部混合，同时保持清晰的区域。通过创建遮挡场景来增强模型的鲁棒性。
        # mosaic=0.8        # 将四个训练图像组合成一个，模拟不同的场景组成和物体交互。对于复杂的场景理解非常有效。
    )

if __name__ == "__main__":
    # main_pt()
    # main_yaml()
    main_yaml1()