import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PETBase(Dataset):
    def __init__(self,
                 txt_file,
                 attn_root,
                 no_attn_root,
                 fuse_attn_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 apply_mask=True,
                 debug=False,
                 debug_dir="debug"
                 ):
        """
        多通道npy格式PET数据加载数据集类
        
        Args:
            txt_file: 数据路径列表文件
            attn_root: 注意力数据根目录
            no_attn_root: 无注意力数据根目录
            fuse_attn_root: 融合注意力数据根目录
            size: 图像尺寸
            interpolation: 插值方式
            flip_p: 水平翻转概率
            apply_mask: 是否应用多通道处理
            debug: 是否启用调试模式
            debug_dir: 调试输出目录
        """
        self.data_paths = txt_file
        self.attn_root = attn_root
        self.no_attn_root = no_attn_root
        self.fuse_attn_root = fuse_attn_root
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "attn_path_": [os.path.join(self.attn_root, l) for l in self.image_paths],
            "no_attn_path_": [os.path.join(self.no_attn_root, l) for l in self.image_paths],
            "fuse_attn_path_": [os.path.join(self.fuse_attn_root, l) for l in self.image_paths],
        }

        self.size = size
        self.interpolation = interpolation
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
        # 调试参数
        self.apply_mask = apply_mask
        self.debug = debug
        self.debug_dir = debug_dir
        
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
        # 加载npy格式数据（直接保持原始float类型）
        attn_image = np.load(example["attn_path_"])
        no_attn_image = np.load(example["no_attn_path_"])
        fuse_attn_image = np.load(example["fuse_attn_path_"])
        
        # 确保数据为3通道（针对PET数据的通道适配）
        def ensure_3ch(img):
            # 若为单通道，扩展为3通道（保持float32类型）
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
            # 确保数据类型为float32
            return img.astype(np.float32)
        
        attn_image = ensure_3ch(attn_image)
        no_attn_image = ensure_3ch(no_attn_image)
        fuse_attn_image = ensure_3ch(fuse_attn_image)
        
        # 图像预处理和中心裁剪（全程使用float32）
        def process_image(img):
            # 移除uint8转换，直接使用float32处理
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            # 中心裁剪
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            return img.astype(np.float32)  # 明确保持float32
        
        attn_image = process_image(attn_image)
        no_attn_image = process_image(no_attn_image)
        fuse_attn_image = process_image(fuse_attn_image)
        
        # 应用多通道处理
        if self.apply_mask:
            # 创建三通道结果（保持float32）
            combined_image = np.zeros_like(attn_image, dtype=np.float32)
            
            # 第一通道：使用注意力图像
            combined_image[:, :, 0] = attn_image[:, :, 0]
            
            # 第二通道：使用无注意力图像
            combined_image[:, :, 1] = no_attn_image[:, :, 1]
            
            # 第三通道：使用融合注意力图像
            combined_image[:, :, 2] = fuse_attn_image[:, :, 2]
            
            image = combined_image
        else:
            image = attn_image  # 如果不应用多通道处理，默认使用注意力图像
        
        # 归一化到[-1, 1]（保持float32）
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["image"] = image
        
        return example


class PET_ab_Train(PETBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/train.txt",
            attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/train/18f_DOPA_sino",
            no_attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/train/18F_FDG_sino",
            fuse_attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/train/Dual_tracer_sino",** kwargs
        )   


class PET_ab_Val(PETBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/val.txt",
            attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/val/18f_DOPA_sino",
            no_attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/val/18F_FDG_sino",
            fuse_attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_brain/val/Dual_tracer_sino",** kwargs
        )