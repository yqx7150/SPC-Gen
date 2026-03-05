import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        image = Image.open(example["file_path_"])
        
        # --------------------------
        # 修复1：处理非RGB图像的维度异常（核心解包错误）
        # --------------------------
        if not image.mode == "RGB":
            # 强制转为单通道灰度图（避免LA/RGBA等多通道模式）
            gray_image = image.convert("L")  # 关键：转为纯单通道，模式变为"L"
            # 转为numpy数组（此时必为2维 (height, width)）
            gray_array = np.array(gray_image)
            # 去除冗余维度（如(512,512,1)这类极端情况）
            gray_array = np.squeeze(gray_array)
            # 最终校验：确保是2维，避免解包失败
            if len(gray_array.shape) != 2:
                raise ValueError(
                    f"非RGB图像维度异常！文件：{example['file_path_']} "
                    f"原始模式：{image.mode} 处理后shape：{gray_array.shape}（预期2维）"
                )
            
            # 保留原有灰度转RGB逻辑
            height, width = gray_array.shape  # 现在不会报错
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_array[:, :, 0] = gray_array
            image = Image.fromarray(rgb_array, mode='RGB')
        
        # 以下保留你原有的所有逻辑
        if i == 0:
            image.save("./0.png")
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        
        # --------------------------
        # 修复2：维度索引错误（原[:, [1,2], :] → 正确[:, :, [1,2]]）
        # --------------------------
        image[:, :, [1, 2]] = 0  # 修正后：对H、W维度的所有像素，将C通道的1、2置0
        
        example["image"] = image
        return example


class PET_SIN_512_train_brain(PETBase):
    def __init__(self, **kwargs):
        if 'config' in kwargs:
            kwargs.pop('config')
        super().__init__(txt_file="/home/b109/Desktop/zn/ALL_PET/txt_lesion/lung_train_clean.txt", data_root="/home/b109/Desktop/zn/CT/data_clean/lung_img_data/images", **kwargs)

class PET_SIN_512_val_brain(PETBase):
    def __init__(self, **kwargs):
        if 'config' in kwargs:
            kwargs.pop('config')
        super().__init__(txt_file="/home/b109/Desktop/zn/ALL_PET/txt_lesion/lung_val.txt", data_root="/home/b109/Desktop/zn/CT/data_clean/lung_img_data/val", **kwargs)