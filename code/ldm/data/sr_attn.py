import os, cv2, albumentations, PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageNetSR(Dataset):
    def __init__(self, txt_file, hr_root, lr_root, size=None,
                 min_crop_f=0.5, max_crop_f=1.0, random_crop=True):
        with open(txt_file, "r") as f:
            self.image_paths = f.read().splitlines()

        self.hr_root = hr_root
        self.lr_root = lr_root
        self.size = size
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.center_crop = not random_crop

        assert size is not None
        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, 
            interpolation=cv2.INTER_AREA
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        relative_path = self.image_paths[i]
        hr_path = os.path.join(self.hr_root, relative_path)
        lr_path = os.path.join(self.lr_root, relative_path)

        # 加载图像并转换为numpy数组
        hr_image = np.array(Image.open(hr_path).convert("RGB"))
        lr_image = np.array(Image.open(lr_path).convert("RGB"))

        # 确保数据类型为float32
        hr_image = hr_image.astype(np.float32)
        lr_image = lr_image.astype(np.float32)

        # 裁剪处理
        min_side_len = min(hr_image.shape[:2])
        crop_side_len = int(min_side_len * np.random.uniform(
            self.min_crop_f, 
            self.max_crop_f
        ))

        cropper = (albumentations.CenterCrop if self.center_crop 
                  else albumentations.RandomCrop)(
            height=crop_side_len, 
            width=crop_side_len
        )

        # # 应用裁剪
        # cropped = cropper(image=hr_image, LR_image=lr_image)
        # hr_image = cropped["image"]
        # lr_image = cropped["LR_image"]

        # # 缩放图像
        # resized = self.image_rescaler(image=hr_image, LR_image=lr_image)
        # hr_image = resized["image"]
        # lr_image = resized["LR_image"]

        # 只保留R通道，G和B置为0
        lr_image[:, :, 1:] = 0  # 将G和B通道设为0

        hr_image[:, :, 1:] = 0  # 将G和B通道设为0


        # 归一化到[-1, 1]范围
        hr_image = (hr_image / 127.5) - 1.0
        lr_image = (lr_image / 127.5) - 1.0

        example = {
            "image": hr_image,
            "LR_image": lr_image,
            "relative_file_path_": relative_path,
            "file_path_": hr_path
        }

        return example


class MySRTrain(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/chenkang/super_pet/attn_data/train/2025_GaoYang/train_labels.txt",
            hr_root="/mnt/D/chenkang/super_pet/attn_data/train/2025_GaoYang/pet_attn_png",
            lr_root="/mnt/D/chenkang/super_pet/attn_data/train/2025_GaoYang/pet_no_attn_png",
            **kwargs
        )


class MySRValidation(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="/mnt/D/chenkang/super_pet/attn_data/val/2025_YangJiaHu/val_labels.txt",
            hr_root="/mnt/D/chenkang/super_pet/attn_data/val/2025_YangJiaHu/pet_attn_png",
            lr_root="/mnt/D/chenkang/super_pet/attn_data/val/2025_YangJiaHu/pet_no_attn_png",
            **kwargs
        )
