import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob
from skimage.transform import radon
from PIL import Image
import matplotlib.pyplot as plt
import odl
import os
def pad_images_with_average(input_npz_path, output_npz_path, n):
        # 加载 npz 文件
        data = np.load(input_npz_path)
        # 假设图像数据的键名为 'images'，可根据实际情况修改
        images = data['arr_0']
        # 处理每个图像
        processed_images = []
        for image in images:
            # 只保留第一个通道，转换为灰度图像
            gray_image = image[:, :, 0]
            #gray_image = image[:, :, 0]；为0时是第一个通道，为1时是第二个通道
            # gray_image[gray_image < 10] = 0
            # 检查图像宽度是否足够
            width = gray_image.shape[1]
            if width < 2 * n:
                raise ValueError("图像宽度小于 2n，无法完成补齐操作。")
            # 计算 n 到 2n 列的平均值
            average_values = np.mean(gray_image[:, n:2 * n], axis=1, keepdims=True)
            # 用平均值补齐前 n 列
            gray_image[:, :n] = average_values
            processed_images.append(gray_image)
            k = Image.fromarray(gray_image, mode="L")
            k.save(f"{random.randint(1, 10)}.jpg")

        # 将处理后的图像列表转换为 numpy 数组
        image = np.array(processed_images)

        # 保存处理后的图像序列到新的 npz 文件
        np.savez(output_npz_path, images=image)


if __name__ == "__main__":
    root = "/mnt/D/ywy/ALL_PET/SD_dataests/unconditional/samples/00549399/2026-03-04-10-40-49/numpy"
    input_npz_path = '1000x512x512x3-samples.npz'
    output_npz_path = '1000x512x512x1-samples.npz'
    #output_npz_path = '100x512x512x1-samples.npz'这是单数据训练生成；output_npz_path = '100x512x512x22-samples.npz'这是多数据配对训练生成，11代表第一个通道的，22代表第二个通道，33代表第三个通道
    n = 10  # 要补齐的列数
    pad_images_with_average(os.path.join(root, input_npz_path), os.path.join(root, output_npz_path), n)
    # 处理之后的数据
    out_images = np.load(os.path.join(root, output_npz_path))['images']
    # 读取投影域图片
    
    # 保存反投影后的图片
    output_dir = os.path.join(root, "last_image") # 输出目录也相应修改
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir)
    plt.savefig(output_path)