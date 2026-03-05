import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom
from skimage.transform import radon
import matplotlib.pyplot as plt

class PETBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 img_size=512,       # 掩码图像尺寸
                 block_size=64,      # 掩码块大小
                 apply_mask=True,
                 debug=False,        # 调试模式
                 debug_dir="debug"   # 调试输出目录
                 ):
        """
        多通道掩码应用数据集类（使用Radon域掩码）
        
        Args:
            txt_file: 图像路径列表文件
            data_root: 数据根目录
            size: 图像尺寸
            interpolation: 插值方式
            flip_p: 水平翻转概率
            img_size: 掩码基础尺寸
            block_size: 掩码块大小
            apply_mask: 是否应用掩码
            debug: 是否启用调试模式
            debug_dir: 调试输出目录
        """
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
        # 掩码生成参数
        self.img_size = img_size
        self.block_size = block_size
        self.theta = np.linspace(0., 180., max(img_size, img_size), endpoint=False)
        
        # 调试参数
        self.apply_mask = apply_mask
        self.debug = debug
        self.debug_dir = debug_dir
        
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

    def _generate_radon_mask(self):
        """生成Radon域掩码"""
        # 随机选择Mask位置
        row = np.random.randint(0, self.img_size - self.block_size + 1)
        col = np.random.randint(0, self.img_size - self.block_size + 1)
        
        # 构建图像域Mask：内部1（白色），外部0（黑色）
        mask_img = np.zeros((self.img_size, self.img_size))
        mask_img[row:row + self.block_size, col:col + self.block_size] = 1.0
        
        # 计算Radon域掩膜
        radon_mask = radon(mask_img, theta=self.theta)
        
        # 归一化Radon掩码到[0,1]
        radon_mask = (radon_mask - radon_mask.min()) / (radon_mask.max() - radon_mask.min() + 1e-8)
        
        # 调整维度以匹配图像
        if radon_mask.shape[0] > self.img_size:
            radon_mask = radon_mask[:self.img_size, :self.img_size]
        radon_mask = radon_mask.T  # 转置以匹配图像坐标系
        
        return radon_mask

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        
        # 处理灰度图像转为RGB
        if not image.mode == "RGB":
            gray_array = np.array(image)
            height, width = gray_array.shape
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_array[:, :, 0] = gray_array  # 第一通道保留灰度 gray_array[512x512x1] rgb_array[512x512x3] 只不过将第一个通道为1，后面两个通道不复制
            rgb_array[:, :, 1] = gray_array 
            rgb_array[:, :, 2] = gray_array 
            image = Image.fromarray(rgb_array, mode='RGB')
        
        # 图像预处理
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        # image = Image.fromarray(img)
        image = img.copy().astype(np.float64)
        
        # if self.size is not None:
        #     image = image.resize((self.size, self.size), resample=self.interpolation)
        # image = self.flip(image)
        # image = np.array(image).astype(np.uint8)
        #image = (image / 127.5 - 1.0).astype(np.float32)  # 归一化到[-1, 1]
        
        # 应用多通道掩码策略
        if self.apply_mask:
            # 生成Radon域掩码
            radon_mask = self._generate_radon_mask()
            
            # 调整掩码尺寸匹配图像
            # if radon_mask.shape != image.shape[:2]:
            #     zoom_factor = (image.shape[0]/radon_mask.shape[0], image.shape[1]/radon_mask.shape[1])
            #     radon_mask = zoom(radon_mask, zoom_factor, order=1)  # 双线性插值
            
            # 创建用于保存各通道结果的数组
            masked_image = image.copy()
            
            # 第一通道：不应用掩码（保持原始）
            # 第二通道：乘以掩码
            masked_image[:, :, 1] *= (1.0 - radon_mask)
            
            # 第三通道：乘以1-掩码
            masked_image[:, :, 2] *= radon_mask
            
            image = masked_image
        
        # 调试模式下保存各通道图像
        if self.debug and i < 5:  # 只保存前5个样本用于调试
            # 保存原始图像
            original_img = ((image + 1.0) * 127.5).astype(np.uint8)
            Image.fromarray(original_img).save(f"{self.debug_dir}/sample_{i}_original.png")
            
            # 保存各通道
            for c in range(3):
                channel_data = image[:, :, c]
                channel_img = ((channel_data + 1.0) * 127.5).astype(np.uint8)
                Image.fromarray(channel_img).save(f"{self.debug_dir}/sample_{i}_channel_{c}.png")
        
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["image"] = image
        
        return example


class PET_SIN_512_train_brain(PETBase):
    def __init__(self, **kwargs):
        if 'config' in kwargs:
            kwargs.pop('config')
        super().__init__(txt_file="/mnt/D/chenkang/high_res_PET/PET_brain/train_sin_png.txt", data_root="/mnt/D/chenkang/high_res_PET/PET_brain/train_sin_png", **kwargs)

class PET_SIN_512_val_brain(PETBase):
    def __init__(self, **kwargs):
        if 'config' in kwargs:
            kwargs.pop('config')
        super().__init__(txt_file="/mnt/D/chenkang/high_res_PET/PET_brain/val_sin_png.txt", data_root="/mnt/D/chenkang/high_res_PET/PET_brain/val_sin_png", **kwargs)


