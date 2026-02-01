from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from pathlib import Path
import os

# GraspNet 数据集根目录
dataset_root = Path(r"E:\yan3up\Code\Py\datasets\GraspNet-1Billion")

# 遍历所有 scene_xxxx 文件夹
scenes = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("scene_")])

print(f"找到 {len(scenes)} 个场景，开始批量转换...")

for scene_dir in scenes:
    masks_dir = scene_dir / "realsense" / "label"
    output_dir = scene_dir / "realsense" / "label_yolo"
    
    if not masks_dir.exists():
        print(f"跳过 {scene_dir.name}: 找不到 label 目录")
        continue

    print(f"正在处理: {scene_dir.name} ...")
    
    try:
        # 确保输出目录存在 (虽然 converter 已经加了创建逻辑，但双重保险无害)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        convert_segment_masks_to_yolo_seg(
            masks_dir=str(masks_dir),
            output_dir=str(output_dir),
            classes=256  # 稍微调大了 classes 以防万一
        )
    except Exception as e:
        print(f"处理 {scene_dir.name} 时出错: {e}")

print("所有场景处理完毕！")