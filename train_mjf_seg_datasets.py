from ultralytics import YOLO

def main_pt():
    # 1. 加载模型
    model = YOLO("yolo11s-seg.pt")

    # 2. 开始训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V2/dataset.yaml",
        epochs=300,
        imgsz=640,
        batch=32,
        device=0,
        name="train-mjf_seg_datasets_V2-yolo11s_seg", # 实验名称
        workers=8,   # 数据加载线程数
        exist_ok=True, # 允许覆盖现有的 project/name 目录

        # === 数据增强 (加大难度) ===
        copy_paste=0.2,
        mixup=0.1,        #【新增】开启 Mixup
        degrees=10.0,       #【新增】增加旋转范围
        cutmix=0.1         #【新增】开启 CutMix
    )

def main_yaml():
    # step 1: 实例化模型
    # 注意：这里必须填你的【YAML文件】，而不是 .pt 文件
    # 这告诉 YOLO：先按这个 YAML 搭积木，把网络层建好（此时参数是随机的）
    model = YOLO("/home/aa205/MJF/ultralytics/ultralytics/cfg/models/11/yolo11s-seg-DySample.yaml")

    # step 2: 加载预训练权重 (迁移学习的关键)
    # 这告诉 YOLO：去官方权重里找名字一样的层，把参数复制过来
    # 控制台会显示 "Transferred X/Y items..."
    model.load("yolo11s-seg.pt")

    # step 3: 开始训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V2/dataset.yaml",
        epochs=300,
        # patience=50,       #【新增】早停机制：50轮不涨就停
        imgsz=640,
        batch=32,
        device=0,
        name="train-mjf_seg_datasets_V2-yolo11s_seg_DySample-deterministic_False",  # 实验名称
        workers=8,  # 数据加载线程数
        exist_ok=True,  # 允许覆盖现有的 project/name 目录
        deterministic=False, # 为True，强制使用确定性算法

        # # === 优化器正则化 ===
        # optimizer='AdamW',
        # weight_decay=0.001, #【修改】加大权重衰减 (默认0.0005)
        # dropout=0.1,        #【新增】增加 Dropout 防止过分依赖某些特征

        # === 数据增强 (加大难度) ===
        copy_paste=0.2,
        mixup=0.1,        #【新增】开启 Mixup
        degrees=10.0,       #【新增】增加旋转范围
        cutmix=0.1         #【新增】开启 CutMix
    )

if __name__ == "__main__":
    # main_pt()
    main_yaml()