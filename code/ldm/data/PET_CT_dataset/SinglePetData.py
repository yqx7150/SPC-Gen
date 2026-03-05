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

        # 确保图像为灰度模式
        if not image.mode == "L":
            image = image.convert("L")
        # 默认使用 score-sde 预处理
        img = np.array(image).astype(np.uint8)
        # 裁剪为正方形
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img, mode="L")  # 确保图像仍为灰度模式
        # 调整大小
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # 水平翻转
        image = self.flip(image)

        # 转换为 numpy 数组并归一化
        image = np.array(image).astype(np.float32)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example



class PET_SIN_512_train(PETBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/mnt/D/yxb/pet/part_train.txt", data_root="/mnt/D/yxb/pet/pet_sin_part_train", **kwargs)

class PET_SIN_512_val(PETBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/mnt/D/yxb/pet/part_val.txt", data_root="/mnt/D/yxb/pet/pet_sin_part_val", **kwargs)

# class PET_SIN_512_train(PETBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="/mnt/D/chenkang/CT_512_sin_png/train.txt", data_root="/mnt/D/chenkang/CT_512_sin_png/train_img", **kwargs)

# class PET_SIN_512_val(PETBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="/mnt/D/chenkang/CT_512_sin_png/val.txt", data_root="/mnt/D/chenkang/CT_512_sin_png/val_img", **kwargs)
