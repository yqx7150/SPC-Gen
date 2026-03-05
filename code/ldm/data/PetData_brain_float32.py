import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision.transforms.functional import resize


class PETBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        file_path = example["file_path_"]
        
        # 读取NPY文件（假设NPY文件存储的是单通道医学图像数据）
        image_np = np.load(file_path)  # shape: (H, W) 或 (H, W, 1)
        
        # 确保数据维度正确（转为单通道）
        if image_np.ndim == 3 and image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)  # 去除通道维度
        
        # 归一化到 [0, 255] 范围（根据实际数据分布调整）
        # 这里假设原始数据已经是合理范围，如果不是需要根据实际情况修改
        #image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8) * 255).astype(np.uint8)
        image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8) * 255).astype(np.float32)

        
        # 转换为RGB格式（保持与原有代码兼容）
        h, w = image_np.shape
        #gb_array = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_array = np.zeros((h, w, 3), dtype=np.float32)
        rgb_array[:, :, 0] = image_np  # 仅使用R通道
        
        # 裁剪为正方形
        crop = min(rgb_array.shape[0], rgb_array.shape[1])
        h, w = rgb_array.shape[0], rgb_array.shape[1]
        rgb_array = rgb_array[(h - crop) // 2:(h + crop) // 2,
                             (w - crop) // 2:(w + crop) // 2]
        
        # 转换为Image对象并调整大小
        #mage = Image.fromarray(rgb_array)

        # PIL处理时临时转为uint8，避免报错
        # image = Image.fromarray(rgb_array.astype(np.uint8))
        # if self.size is not None:
        #     image = image.resize((self.size, self.size), resample=self.interpolation)

        # 新代码：
        if self.size is not None:
            # 转换为torch张量（形状：[H, W, C] -> [C, H, W]）
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()
            # 使用torchvision的resize（支持浮点输入）
            rgb_tensor = resize(
                rgb_tensor,
                (self.size, self.size),
                interpolation=self.interpolation,
                antialias=True  # 抗锯齿（可选）
            )
            # 转回numpy数组（形状：[C, H, W] -> [H, W, C]）
            rgb_array = rgb_tensor.permute(1, 2, 0).numpy()

        # 数据增强（水平翻转）：改用torch实现
        image_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()
        image_tensor = self.flip(image_tensor)  # 此时flip作用于张量
        image = image_tensor.permute(1, 2, 0).numpy()
        
        # # 数据增强（水平翻转）
        # image = self.flip(image)


        
        # 转换为numpy数组并归一化到 [-1, 1]
        # image = np.array(image).astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        image = np.array(image).astype(np.float32)  # 保持float32
        image = (image / 127.5 - 1.0)  # 已为float32，无需再次转换
        
        # 保持原有代码的通道处理逻辑
        image[:, [1, 2], :] = 0  # 仅保留R通道数据
        
        example["image"] = image
        return example


class PET_SIN_512_train_brain(PETBase):
    def __init__(self, **kwargs):
        if 'config' in kwargs:
            kwargs.pop('config')
        # 注意：确保txt文件中存储的是NPY文件的路径（例如xxx.npy）
        super().__init__(txt_file="/mnt/D/ywy/ALL_PET/ywy_dataests/brain_npy/train.txt", 
                         data_root="/mnt/D/ywy/ALL_PET/ywy_dataests/brain_npy/train",** kwargs)

class PET_SIN_512_val_brain(PETBase):
    def __init__(self, **kwargs):
        if 'config' in kwargs:
            kwargs.pop('config')
        # 注意：确保txt文件中存储的是NPY文件的路径（例如xxx.npy）
        super().__init__(txt_file="/mnt/D/ywy/ALL_PET/ywy_dataests/brain_npy/val.txt", 
                         data_root="/mnt/D/ywy/ALL_PET/ywy_dataests/brain_npy/val",** kwargs)