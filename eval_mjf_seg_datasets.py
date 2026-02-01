from ultralytics import YOLO
import os


def main():
    # 1. 确认模型路径
    model_path = "/home/aa205/MJF/ultralytics/runs/segment/train-mjf_seg_datasets_V2-yolo11s_seg_DySample/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    model = YOLO(model_path)

    print("开始在测试集 (Test Set) 上进行评估...")

    # 2. 核心评估代码
    metrics = model.val(
        data="/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V2/dataset.yaml",
        split="test",  # 确定用于验证的数据集分割（val, test或 train）
        batch=16,
        imgsz=640,
        device=0,
        plots=True,  # 生成并保存预测与真实值的对比图、混淆矩阵和 PR 曲线
        name="val_train-mjf_seg_datasets_V2-yolo11s_seg_DySample",
        exist_ok=True  # 允许覆盖现有的 project/name 目录
    )

    # 3. 输出核心指标 (YOLO会自动打印详细表格，这里只需提炼关键数据)
    print("\n" + "=" * 40)
    print("测试集最终评估结果 (Test Set Results):")
    # Mask Metrics
    print(f"Mask Mean Precision : {metrics.seg.mp:.4f}")
    print(f"Mask Mean Recall    : {metrics.seg.mr:.4f}")
    print(f"Mask mAP50         : {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95      : {metrics.seg.map:.4f}")
    print("-" * 20)
    # Box Metrics (Optional but useful for context)
    print(f"Box  mAP50         : {metrics.box.map50:.4f}")
    print(f"Box  mAP50-95      : {metrics.box.map:.4f}")
    print("=" * 40)
    print(f"详细评估图表已保存在: {metrics.save_dir}")


if __name__ == "__main__":
    main()