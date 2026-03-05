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
            rgb_array[:, :, 0] = gray_array  # 第一通道保留灰度
            
            image = Image.fromarray(rgb_array, mode='RGB')
        
        # 图像预处理
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        image = img.copy().astype(np.float64)
        
        # 缩放图像到256×256
        image_256 = self._resize_image(image, 256)
        
        # 生成掩码
        if self.apply_mask:
            radon_mask = self._generate_radon_mask()
            # 缩放到256×256
            radon_mask_256 = self._resize_image(radon_mask, 256)
        else:
            # 如果不应用掩码，创建全1的掩码
            radon_mask_256 = np.ones((256, 256))
        
        # 创建四宫格图像
        combined_image = np.zeros((512, 512, 3), dtype=np.float64)
        
        # 左上角：原图
        combined_image[0:256, 0:256] = image_256
        
        # 右上角：(1-mask)×原图
        masked_image_1 = image_256 * (1.0 - radon_mask_256)[..., np.newaxis]
        combined_image[0:256, 256:512] = masked_image_1
        
        # 左下角：mask×原图
        masked_image_2 = image_256 * radon_mask_256[..., np.newaxis]
        combined_image[256:512, 0:256] = masked_image_2
        
        # 右下角：原图
        combined_image[256:512, 256:512] = image_256
        
        image = combined_image
        
        
        
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image[:, [1, 2], :] = 0 #读数据
        # image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        example["image"] = image
        
        return example
    
    def _resize_image(self, image, target_size):
        """将图像缩放到指定尺寸"""
        if len(image.shape) == 2:  # 处理掩码（单通道）
            image_pil = Image.fromarray(image)
        else:  # 处理RGB图像
            image_pil = Image.fromarray(image.astype(np.uint8))
        
        resized = image_pil.resize((target_size, target_size), resample=self.interpolation)
        
        if len(image.shape) == 2:
            return np.array(resized, dtype=np.float64)
        else:
            return np.array(resized, dtype=np.float64)


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