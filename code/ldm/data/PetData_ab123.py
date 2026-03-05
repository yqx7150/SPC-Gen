import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


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
        多通道数据加载数据集类
        
        Args:
            txt_file: 图像路径列表文件
            attn_root: 注意力图像根目录（原data_root）
            no_attn_root: 无注意力图像根目录
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
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
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
        
        # 加载注意力图像
        attn_image = Image.open(example["attn_path_"])
        # 加载无注意力图像
        no_attn_image = Image.open(example["no_attn_path_"])

        fuse_attn_image = Image.open(example["fuse_attn_path_"])
        
        # 确保所有图像都是RGB模式
        def ensure_rgb(img):
            if not img.mode == "RGB":
                gray_array = np.array(img)
                # gray_array = gray_array[:, :, 0]  # 取 R 通道（假设 R=G=B）
                height, width = gray_array.shape
                rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_array[:, :, 0] = gray_array
                rgb_array[:, :, 1] = gray_array
                rgb_array[:, :, 2] = gray_array
                return Image.fromarray(rgb_array, mode='RGB')
            return img
        
        attn_image = ensure_rgb(attn_image)
        no_attn_image = ensure_rgb(no_attn_image)
        fuse_attn_image = ensure_rgb(fuse_attn_image)
        
        # 图像预处理和中心裁剪
        def process_image(img):
            img = np.array(img).astype(np.uint8)
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            return img.astype(np.float64)
        
        attn_image = process_image(attn_image)
        no_attn_image = process_image(no_attn_image)
        fuse_attn_image = process_image(fuse_attn_image)
        
        # 应用多通道处理
        if self.apply_mask:
            # 创建三通道结果
            combined_image = np.zeros_like(attn_image)
            
            # 第一通道：使用注意力图像
            combined_image[:, :, 0] = attn_image[:, :, 0]
            
            # 第二通道：使用无注意力图像
            combined_image[:, :, 1] = no_attn_image[:, :, 1]
            
            # 第三通道：设为0
            combined_image[:, :, 2] = fuse_attn_image[:, :, 2]
            
            image = combined_image
        else:
            image = attn_image  # 如果不应用多通道处理，默认使用注意力图像
        
        # 归一化到[-1, 1]
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["image"] = image
        
        return example


class PET_ab_Train(PETBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/train.txt",
            attn_root="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/train/18F",
            no_attn_root="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/train/68G",
            fuse_attn_root="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/train/Fuse",
            #txt_file="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/npy_bp_png_500.txt",
            #attn_root="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/F18_npy_bp_png_500",
            #no_attn_root="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/G68_npy_bp_png_500",
            #fuse_attn_root="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/fuse_npy_bp_png_500",
            **kwargs
        )   


class PET_ab_Val(PETBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/val.txt",
            attn_root="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/val/18F_sino_png",
            no_attn_root="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/val/68G_sino_png",
            fuse_attn_root="/mnt/D/ywy/ALL_PET/Dual/chest_transverse/val/Fuse_sino_png",
            #txt_file="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/npy_bp_png_100.txt",
            #attn_root="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/F18_npy_bp_png_100",
            #no_attn_root="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/G68_npy_bp_png_100",
            #fuse_attn_root="/home/b109/Desktop/chenkang/PET_split_data_abdomen/train/fuse_npy_bp_png_100",
            **kwargs
        )