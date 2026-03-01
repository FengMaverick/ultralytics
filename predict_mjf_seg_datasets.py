from ultralytics import YOLO
import os

def main():
    # 1. 确认模型路径 (请替换为你想要使用的已训练好的权重路径)
    model_path = "runs/segment/train-msdyV3-yolo11s_seg/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"警告: 找不到模型文件 {model_path}，请确认路径是否正确。")
        # 即使这里提示警告，为了方便你查看代码结构，仍然继续（实际运行可能会报错）

    # 2. 加载模型
    model = YOLO(model_path)
    print(f"成功加载模型: {model_path}")

    # 3. 设置预测的输入源
    # source 可以是:
    # - 单张图片: "path/to/image.jpg"
    # - 包含多张图片的目录: "path/to/images/"
    # - 视频: "path/to/video.mp4"
    # - USB摄像头或推流: 0 (代表设备 0)
    # 请替换为你要预测的实际图片或文件夹路径。比如我们预测验证集的图像：
    source_path = "/home/aa205/MJF/datasets/Instance_Segment_datasets/mjf_seg_datasets_yolo_V3/images/val"

    print(f"\n开始对目标进行预测 (Inference):\n --> {source_path}")

    # 4. 核心预测代码 (Inference)
    # predictor 会返回一个由对象列表组成的生成器（每个图像一个对象）
    results = model.predict(
        source=source_path, # 预测的数据源
        imgsz=640,          # 图像缩放大小
        conf=0.25,          # 置信度阈值，低于该值的预测对象将被忽略 (默认0.25)
        iou=0.45,           # NMS 交并比阈值，用于去除重复框 (默认0.7)
        device=0,           # 使用 GPU 0, 若使用 CPU 请设为 "cpu"
        save=True,          # 是否保存预测结果（预测出的掩膜和边界框覆盖在图片上）
        save_txt=False,     # 是否将预测的边界框等信息保存为 txt 文件
        save_conf=False,    # 如果保存 txt，是否在 txt 中包含置信度分数
        save_crop=False,    # 是否切割并保存有预测目标的图片块
        name="predict_msdyV3-yolo11s_seg", # 保存结果的文件夹名称
        exist_ok=True       # 允许覆盖现有的 project/name 目录
    )

    # 5. 可选：可以在代码里解析遍历结果对象
    print("\n" + "=" * 40)
    print("预测完成！")
    # 如果你想在代码里对预测结果做进一步处理，比如针对每一张预测通过模型提取出掩码：
    # for result in results:
    #     boxes = result.boxes  # 边界框信息 [xyxy, conf, cls]
    #     masks = result.masks  # 分割掩码 (如果是分割人物)
    #     probs = result.probs  # 分类概率 (如果是图片分类任务)
    #     
    #     # 你甚至可以在这里显式展示或者保存图像
    #     # result.show()  
    #     # result.save(filename="result.jpg") 
    print("=" * 40)
    if hasattr(results, "__len__") and len(results) > 0:
        print(f"预测结果和渲染后的图片已保存在其对应的 run 目录下！(类似 runs/segment/predict_...)")

if __name__ == "__main__":
    main()
