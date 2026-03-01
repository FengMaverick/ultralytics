import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ================= 文件夹路径配置 =================
# 原模型的结果图文件夹
DIR1 = r"e:\yan3up\Code\Py\ultralytics\runs\segment\predict_train-msdyV3-yolo11s_seg"
# 改进后模型的结果图文件夹
DIR2 = r"e:\yan3up\Code\Py\ultralytics\runs\segment\predict_train-msdyV3-yolo11s_seg_C3k2_DCNv4_SPPF_Container"
# 保存你精心挑选的对比图的文件夹
SAVE_DIR = r"e:\yan3up\Code\Py\ultralytics\runs\segment\saved_comparisons"
# ==================================================

class ImageCompareTool:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 模型推理结果可视化对比工具")
        
        # 允许工具自动调整并全屏居中展示
        self.root.geometry("1400x800")
        
        # 1. 获取两个文件夹中共同存在的文件名
        if not os.path.exists(DIR1) or not os.path.exists(DIR2):
            messagebox.showerror("路径错误", "请检查 DIR1 或 DIR2 的路径是否存在！")
            self.root.destroy()
            return

        imgs1 = set([f for f in os.listdir(DIR1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        imgs2 = set([f for f in os.listdir(DIR2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.common_images = sorted(list(imgs1.intersection(imgs2)))
        
        if not self.common_images:
            messagebox.showerror("错误", "两个文件夹中没有找到任何相同文件名的图片！")
            self.root.destroy()
            return
            
        self.current_idx = 0
        
        # 创建保存文件夹
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            
        # 2. 构建 UI
        self.setup_ui()
        self.load_image()
        
    def setup_ui(self):
        # 顶部提示和进度
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.lbl_progress = tk.Label(self.info_frame, text="", font=("微软雅黑", 12, "bold"))
        self.lbl_progress.pack()

        # 中间的图片展示区域
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # 左右分栏
        self.left_frame = tk.Frame(self.img_frame)
        self.left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.right_frame = tk.Frame(self.img_frame)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.lbl_title1 = tk.Label(self.left_frame, text="模型 1 解析图 (Baseline)", font=("微软雅黑", 14), fg="blue")
        self.lbl_title1.pack()
        self.panel1 = tk.Label(self.left_frame, bg="gray")
        self.panel1.pack(expand=True, padx=10, pady=10)
        
        self.lbl_title2 = tk.Label(self.right_frame, text="模型 2 解析图 (改进版)", font=("微软雅黑", 14), fg="red")
        self.lbl_title2.pack()
        self.panel2 = tk.Label(self.right_frame, bg="gray")
        self.panel2.pack(expand=True, padx=10, pady=10)
        
        # 底部操作按钮区域
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        # 按钮居中
        self.center_btn_frame = tk.Frame(self.btn_frame)
        self.center_btn_frame.pack(anchor=tk.CENTER)
        
        self.btn_prev = tk.Button(self.center_btn_frame, text="⬅ 上一组 (Left)", font=("微软雅黑", 12), width=15, height=2, command=self.prev_img)
        self.btn_prev.pack(side=tk.LEFT, padx=30)
        
        self.btn_save = tk.Button(self.center_btn_frame, text="💾 保存并下一组 (Space)", font=("微软雅黑", 12, "bold"), width=30, height=2, bg="#87CEFA", command=self.save_img)
        self.btn_save.pack(side=tk.LEFT, padx=30)
        
        self.btn_next = tk.Button(self.center_btn_frame, text="下一组 ➡ (Right)", font=("微软雅黑", 12), width=15, height=2, command=self.next_img)
        self.btn_next.pack(side=tk.LEFT, padx=30)
        
        # 绑定快捷键，体验更丝滑
        self.root.bind('<Left>', lambda e: self.prev_img())
        self.root.bind('<Right>', lambda e: self.next_img())
        self.root.bind('<space>', lambda e: self.save_img())

    def get_resized_image(self, img_path, target_height=650):
        """加载图片并等比例缩放以适应屏幕"""
        img = Image.open(img_path)
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def load_image(self):
        img_name = self.common_images[self.current_idx]
        
        # 更新文字
        self.lbl_progress.config(text=f"进度: {self.current_idx + 1} / {len(self.common_images)}  |  当前文件名: {img_name}")
        
        path1 = os.path.join(DIR1, img_name)
        path2 = os.path.join(DIR2, img_name)
        
        # 为了保证不撑爆屏幕，这里固定高度为 650 像素（你如果显示器大可以往上调）
        self.photo1 = self.get_resized_image(path1, target_height=650)
        self.panel1.config(image=self.photo1)
        
        self.photo2 = self.get_resized_image(path2, target_height=650)
        self.panel2.config(image=self.photo2)
        
        self.update_buttons()

    def prev_img(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def next_img(self):
        if self.current_idx < len(self.common_images) - 1:
            self.current_idx += 1
            self.load_image()

    def save_img(self):
        img_name = self.common_images[self.current_idx]
        path1 = os.path.join(DIR1, img_name)
        path2 = os.path.join(DIR2, img_name)
        
        # 定义子文件夹路径
        save_dir1 = os.path.join(SAVE_DIR, "model1_baseline")
        save_dir2 = os.path.join(SAVE_DIR, "model2_improved")
        
        # 确保子文件夹存在
        os.makedirs(save_dir1, exist_ok=True)
        os.makedirs(save_dir2, exist_ok=True)
        
        # 保持原文件名，分别保存在各自的子文件夹下
        save_path1 = os.path.join(save_dir1, img_name)
        save_path2 = os.path.join(save_dir2, img_name)
        
        # 复制原图到对应目录
        import shutil
        shutil.copy2(path1, save_path1)
        shutil.copy2(path2, save_path2)
        
        print(f"👍 已分别保存原图到子文件夹:\n   -> {save_path1}\n   -> {save_path2}")
        
        # 保存完之后自动跳到下一张
        if self.current_idx < len(self.common_images) - 1:
            self.current_idx += 1
            self.load_image()
        else:
            messagebox.showinfo("完成", "这是最后一张图片了！")

    def update_buttons(self):
        # 第一张禁用“上一张” ，最后一张禁用“下一张”
        self.btn_prev.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_idx < len(self.common_images) - 1 else tk.DISABLED)

if __name__ == "__main__":
    # 如果运行报错说没有 PIL，请在终端执行： pip install pillow
    root = tk.Tk()
    app = ImageCompareTool(root)
    root.mainloop()
