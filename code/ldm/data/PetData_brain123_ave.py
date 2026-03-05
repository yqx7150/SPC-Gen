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
                 average_channels=True  # 是否对通道求平均
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
            average_channels: 是否对三个通道求平均
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
        
        # 通道掩码策略配置
        self.channel_masks = {
            0: None,        # 第一通道不应用掩码
            1: "mask",      # 第二通道应用原始掩码
            2: "1-mask"     # 第三通道应用1-mask
        }
        
        self.apply_mask = apply_mask
        self.average_channels = average_channels

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
        
        # 调整维度以匹配图像（假设Radon掩码为(H, W)）
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
            rgb_array[:, :, 0] = gray_array  # 第一通道保留灰度
            image = Image.fromarray(rgb_array, mode='RGB')
        
        # 图像预处理
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)
        
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)  # 归一化到[-1, 1]
        
        # 应用多通道掩码策略
        if self.apply_mask:
            # 生成Radon域掩码
            radon_mask = self._generate_radon_mask()
            
            # 调整掩码尺寸匹配图像
            if radon_mask.shape != image.shape[:2]:
                zoom_factor = (image.shape[0]/radon_mask.shape[0], image.shape[1]/radon_mask.shape[1])
                radon_mask = zoom(radon_mask, zoom_factor, order=1)  # 双线性插值
            
            # 对每个通道应用指定的掩码策略
            channel_results = []
            for channel in range(3):
                channel_img = image.copy()
                mask_strategy = self.channel_masks.get(channel, None)
                
                if mask_strategy is not None:
                    # 计算掩码（原始或1-mask）
                    if mask_strategy == "mask":
                        mask_to_apply = radon_mask
                    elif mask_strategy == "1-mask":
                        mask_to_apply = 1.0 - radon_mask
                    
                    # 应用掩码到指定通道
                    channel_img[:, :, channel] *= mask_to_apply
                
                channel_results.append(channel_img)
            
            # 叠加通道并求平均
            if self.average_channels:
                image = np.mean(channel_results, axis=0)
            else:
                # 不平均时，使用第一个通道的结果（可根据需求调整）
                image = channel_results[0]
        
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


