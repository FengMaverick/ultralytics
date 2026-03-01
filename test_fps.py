import torch
import time
from ultralytics import YOLO

def calc_fps(model_path, imgsz=640, num_warmup=10, num_test=100):
    # 1. 加载模型
    model = YOLO(model_path)
    
    # 2. 生成一张全黑的伪造图像 (Dummy Image) 用来测试，这能完全排除硬盘读图的 IO 延迟
    # 注意：YOLO 输入格式是 (批次, 通道, 高, 宽)，RGB图像为 3 通道
    dummy_input = torch.zeros(1, 3, imgsz, imgsz).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📊 开始测试模型: {model_path}")
    print("⏳ 正在预热 (Warm-up GPU)...")
    
    # 3. 预热 (Warm-up)
    # GPU 刚开始跑的时候需要初始化内核，第一批图会特别慢，必须跑几次预热才能测出真实速度
    for _ in range(num_warmup):
        model.predict(dummy_input, device=0, verbose=False)

    print(f"🚀 开始测试纯推理时间 (共 {num_test} 次)...")
    
    # 4. 正式测速
    start_time = time.time()
    for _ in range(num_test):
        # verbose=False 关掉控制台打印，防止拖慢速度
        model.predict(dummy_input, device=0, verbose=False)
        
    end_time = time.time()

    # 5. 计算结果
    total_time = end_time - start_time
    avg_time_per_img_ms = (total_time / num_test) * 1000 # 换算成毫秒(ms)
    fps = 1000 / avg_time_per_img_ms                     # FPS = 1000 / 单张图耗时(ms)
    
    # 6. 计算参数量 (Parameters) 和 计算量 (FLOPs)
    # ultralytics 的 model.info() 会返回 (layers, parameters, gradients, flops)
    try:
        model_info = model.info(verbose=False)
        params_millions = model_info[1] / 1e6  # 转换为百万 (M)
        flops_giga = model_info[3]             # 返回的直接是 GFLOPs
    except:
        # 如果 info 方法有些许变动，手动计算作为备用
        params = sum(p.numel() for p in model.model.parameters())
        params_millions = params / 1e6
        flops_giga = "N/A" # 备用方法不计算 FLOPs

    print(f"✅ 平均耗时: {avg_time_per_img_ms:.2f} ms / 张 (纯向前传播)")
    print(f"✅ 估算 FPS : {fps:.2f} 帧 / 秒")
    print(f"✅ 模型参数量 (Params): {params_millions:.2f} M")
    if flops_giga != "N/A":
        print(f"✅ 模型计算量 (FLOPs): {flops_giga:.2f} G")

if __name__ == "__main__":
    # 把你想测的 best.pt 路径都写在这里循环跑
    models_to_test = [
        "runs/segment/train-msdyV3-yolo11s_seg/weights/best.pt",
        "runs/segment/train-msdyV3-yolo11s_seg_C3k2_DCNv4_V3/weights/best.pt",
        "runs/segment/train-msdyV3-yolo11s_seg_SPPF_Container/weights/best.pt",
        "runs/segment/train-msdyV3-yolo11s_seg_EUCB/weights/best.pt",
        "runs/segment/train-msdyV3-yolo11s_seg_C3k2_DCNv4_SPPF_Container/weights/best.pt",
        "runs/segment/train-msdyV3-yolo11s_seg_C3k2_DCNv4_SPPF_Container_EUCB/weights/best.pt"
    ]
    
    for pt in models_to_test:
        calc_fps(pt, imgsz=640)
