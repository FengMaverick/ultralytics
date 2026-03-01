from ultralytics import YOLO

def train_phase_1():
    print("\n" + "=" * 50)
    print("=== 开始第一阶段：冻结骨干网络 (Freeze Backbone) ===")
    print("目的：保护已有模型的特征提取层，让新加入的 DySample 模块快速学习")
    print("=" * 50)

    # 1. 实例化包含新模块 (C3k2_DCNv4 + SPPF_Container + DySample) 的整个网络结构
    model = YOLO("/home/aa205/MJF/ultralytics/ultralytics/cfg/models/11/yolo11s-seg-C3k2_DCNv4-SPPF_Container-DySample.yaml")
    
    # 2. 加载之前仅包含 C3k2_DCNv4 和 SPPF 的 best.pt 权重。
    #    请将此路径替换为你上一次训练出的最佳权重。
    #    YOLO会按层名和结构自动匹配，缺失的 DySample 会保持随机初始化。
    model.load("runs/segment/train-msdyV3-yolo11s_seg_C3k2_LSBlock/weights/best.pt") # 请根据你的实际路径修改

    # 3. 开始第一阶段的冻结训练
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V3/dataset.yaml",
        epochs=100,           # 阶段一不需要跑满 300 轮，100轮足够让头部和 DySample 收敛
        imgsz=640,
        batch=32,
        device=0,
        name="train_phase1_frozen_DySample", 
        workers=8,
        exist_ok=True,

        # 【核心参数】
        freeze=11,            # 冻结前 11 个模块 (根据你的 yaml，索引 0~10 是 Backbone)
        optimizer='AdamW',    # 沿用你之前效果好的优化器
        lr0=0.000625,         # 沿用 auto 给出的最优基础学习率
        lrf=0.01,             # 最终学习率为初始值的 1%
        warmup_epochs=3.0,    # 给新初始化的 DySample 一个热身时间
        warmup_bias_lr=0.1,   

        # 数据增强：因为主干被冻结，降低增强强度，让 Head 更快抓准目标
        copy_paste=0.0,       # 暂时关闭
        mixup=0.0,            # 暂时关闭
    )
    print("\n✅ 第一阶段训练完成！")
    return "runs/segment/train_phase1_frozen_DySample/weights/best.pt"


def train_phase_2(phase_1_best_pt):
    print("\n" + "=" * 50)
    print("=== 开始第二阶段：全量微调 (Global Fine-tuning) ===")
    print("目的：解冻全局，用极小的学习率和0热身，让全网产生化学反应")
    print("=" * 50)

    # 1. 结构不变
    model = YOLO("/home/aa205/MJF/ultralytics/ultralytics/cfg/models/11/yolo11s-seg-C3k2_DCNv4-SPPF_Container-DySample.yaml")
    
    # 2. 从刚才第一阶段产出的 best.pt 开始加载
    #    此时 Backbone 是以前的高手，Head 和 DySample 也是经过 100 轮特训的高手了
    model.load(phase_1_best_pt)

    # 3. 开始第二阶段的全模型微调
    results = model.train(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V3/dataset.yaml",
        epochs=200,           # 联合精细打磨，补齐剩下的训练轮数 (总共300=100+200)
        imgsz=640,
        batch=32,
        device=0,
        name="train_phase2_finetune_DySample",
        workers=8,
        exist_ok=True,

        # 【极其核弹级参数】
        freeze=None,          # 全部解冻，全员参战
        optimizer='AdamW',    
        lr0=0.0000625,        # 基础学习率降低 10 倍！(0.000625 -> 0.0000625) 保护已有好参数
        lrf=0.01,             
        warmup_epochs=0,      # 取消热身！全网已经是熟手，直接进入细微的梯度下降

        # 数据增强：把增强强度拉满，防止全网联合微调时过拟合
        copy_paste=0.3,  
        mixup=0.1,       
        cutmix=0.1
    )
    print("\n✅ 第二阶段全量微调完成！")


if __name__ == "__main__":
    # 执行两阶段训练
    # 注意：请确保 phase_1 中加载的历史 best.pt 路径是正确的！
    
    # 也可以选择分布运行：如果第一阶段跑到一半断了，接续跑完后，把该行注释掉，
    # 直接运行 train_phase_2 并手动填入第一阶段生成的最佳权重路径即可。
    
    best_pt_from_phase1 = train_phase_1()
    train_phase_2(best_pt_from_phase1)
