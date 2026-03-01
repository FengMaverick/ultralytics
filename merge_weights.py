import torch

def merge_yolo_weights(backbone_pt_path, head_pt_path, save_path="custom_merged_weights.pt"):
    """
    拼接两个 YOLO 权重文件：
    使用 backbone_pt_path 里的前面部分（Backbone）
    使用 head_pt_path 里的后面部分（Neck/Head）
    """
    print("开始进行权重缝合手术...")
        
    # 1. 加载两个权重文件里的模型字典 (.pt 文件本质上是个存了各项配置的字典集)
    # PyTorch 2.6 起，安全机制限制了反序列化第三方类（YOLO 的模型类），这里安全信任来源故加上 weights_only=False
    ckpt_backbone = torch.load(backbone_pt_path, map_location="cpu", weights_only=False)
    ckpt_head = torch.load(head_pt_path, map_location="cpu", weights_only=False)
    
    # YOLO 权重的核心参数存在 'model' 键里面的一个 OrderedDict 中
    state_dict_backbone = ckpt_backbone['model'].state_dict()
    state_dict_head = ckpt_head['model'].state_dict()
    
    # 2. 准备一个新的空字典来装拼接好的权重
    merged_state_dict = {}
    
    # 3. 缝合逻辑
    #    我们知道 YOLO11 的 Backbone 是从 model.0 到 model.10
    #    Neck & Head 从 model.11 开始到底
    keys_transferred = 0
    for key in state_dict_backbone.keys(): # 遍历所有层的名字，比如 "model.0.conv.weight"
        # 提取层的数字 ID (例如 "model.0.conv" 提取出 0)
        try:
            layer_id = int(key.split('.')[1]) 
        except ValueError:
            # 有些键可能不带层号，直接按原样处理或跳过，YOLO 里极少
            continue
            
        if layer_id <= 10:
            # 属于 Backbone (0~10层)，使用 best.pt 的参数
            merged_state_dict[key] = state_dict_backbone[key].clone()
            keys_transferred += 1
        else:
            # 属于 Neck / Head (>=11层)，使用官方 yolo11s-seg.pt 的参数
            # 前提是官方权重里有这个键，且 shape 一致。你的 DySample 因为官方里没有，这里自然就加载不进来
            if key in state_dict_head and state_dict_head[key].shape == state_dict_backbone[key].shape:
                merged_state_dict[key] = state_dict_head[key].clone()
                keys_transferred += 1
            else:
                # 如果官方权重里没有，或者 shape 变了（这就是为什么需要 DySample 去随机初始化）
                # 我们就不把它加进 merged_state_dict，YOLO 加载时会自动将其随机初始化
                pass 
                
    # 4. 把拼接好的字典塞回结构里并保存
    ckpt_backbone['model'].load_state_dict(merged_state_dict, strict=False)
    torch.save(ckpt_backbone, save_path)
    
    print(f"✅ 缝合成功！共转移了 {keys_transferred} 个张量，已保存至: {save_path}")
    return save_path

# 执行缝合 (路径请换成你真实的)
merge_yolo_weights(r"E:\yan3up\Code\Py\ultralytics\runs\segment\train-msdyV3-yolo11s_seg_C3k2_DCNv4_SPPF_Container\weights\best.pt", r"E:\yan3up\Code\Py\ultralytics\yolo11s-seg.pt")
