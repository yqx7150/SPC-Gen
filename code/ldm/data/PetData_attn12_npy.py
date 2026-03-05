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
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 apply_mask=True,
                 debug=False,
                 debug_dir="debug"
                 ):
        self.data_paths = txt_file
        self.attn_root = attn_root
        self.no_attn_root = no_attn_root
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "attn_path_": [os.path.join(self.attn_root, l) for l in self.image_paths],
            "no_attn_path_": [os.path.join(self.no_attn_root, l) for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
        self.apply_mask = apply_mask
        self.debug = debug
        self.debug_dir = debug_dir
        
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
        # 加载npy文件
        attn_image = np.load(example["attn_path_"])
        no_attn_image = np.load(example["no_attn_path_"])
        
        # 确保图像为3通道格式
        def ensure_rgb(arr):
            if arr.ndim == 2:  # 二维灰度图 (H, W)
                height, width = arr.shape
                rgb_arr = np.zeros((height, width, 3), dtype=np.float32)
                rgb_arr[:, :, 0] = arr
                rgb_arr[:, :, 1] = arr
                return rgb_arr
            elif arr.ndim == 3 and arr.shape[2] == 1:  # 单通道 (H, W, 1)
                gray_arr = arr[:, :, 0]
                height, width = gray_arr.shape
                rgb_arr = np.zeros((height, width, 3), dtype=np.float32)
                rgb_arr[:, :, 0] = gray_arr
                rgb_arr[:, :, 1] = gray_arr
                return rgb_arr
            elif arr.ndim == 3 and arr.shape[2] == 3:  # 已为RGB
                return arr.astype(np.float32)
            else:  # 其他情况取第一通道
                gray_arr = arr[:, :, 0].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
                height, width = gray_arr.shape[:2]
                rgb_arr = np.zeros((height, width, 3), dtype=np.float32)
                rgb_arr[:, :, 0] = gray_arr
                rgb_arr[:, :, 1] = gray_arr
                return rgb_arr
        
        # 处理npy数组为RGB格式
        attn_image = ensure_rgb(attn_image)
        no_attn_image = ensure_rgb(no_attn_image)
        
        # 图像预处理和中心裁剪（增加临时uint8转换）
        def process_image(arr):
            # 归一化到 [0, 255] 范围（保持float32精度）
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max != arr_min:
                arr = ((arr - arr_min) / (arr_max - arr_min + 1e-8) * 255).astype(np.float32)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)  # 处理极端情况
            
            # 中心裁剪
            crop = min(arr.shape[0], arr.shape[1])
            h, w = arr.shape[0], arr.shape[1]
            arr = arr[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            
            # 转换为PIL Image时临时转为uint8，避免报错
            image_pil = Image.fromarray(arr.astype(np.uint8))
            
            # 调整大小（如果需要）
            if self.size is not None:
                image_pil = image_pil.resize((self.size, self.size), resample=self.interpolation)
            
            # 转回float32格式进行后续处理
            return np.array(image_pil).astype(np.float32)
        
        attn_image = process_image(attn_image)
        no_attn_image = process_image(no_attn_image)
        
        # 应用多通道处理
        if self.apply_mask:
            combined_image = np.zeros_like(attn_image)
            combined_image[:, :, 0] = attn_image[:, :, 0]
            combined_image[:, :, 1] = no_attn_image[:, :, 1]
            image = combined_image
        else:
            image = attn_image
        
        # 数据增强（水平翻转）- 增加临时uint8转换
        image = self.flip(Image.fromarray(image.astype(np.uint8)))
        image = np.array(image).astype(np.float32)
        
        # 归一化到[-1, 1]
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["image"] = image
        
        return example


class PET_attn_Train(PETBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/ywy/ALL_PET/ywy_dataests/Self-attn/train.txt",
            attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Self-attn/train/after1_sino",
            no_attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Self-attn/train/before1_sino",** kwargs
        )   


class PET_attn_Val(PETBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/ywy/ALL_PET/ywy_dataests/Self-attn/val.txt",
            attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Self-attn/val/after",
            no_attn_root="/mnt/D/ywy/ALL_PET/ywy_dataests/Self-attn/val/before",** kwargs
        )