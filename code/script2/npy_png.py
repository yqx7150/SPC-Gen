import os
import numpy as np
from PIL import Image

# 输入文件夹路径（指定的npy文件所在目录）
input_folder = '/mnt/D/ywy/ALL_PET/SD_dataests/unconditional/samples/00549399/2026-03-04-10-40-49/numpy/1000x512x512x1-samples'
#input_folder = '/mnt/D/ywy/ALL_PET/pet_brain_out_ddpm_3/unconditional/samples/00549399/2025-09-26-10-21-57/numpy/100x512x512x1-samples_resino'
# 输出文件夹路径（转换后的png文件存放目录）
output_folder = '/mnt/D/ywy/ALL_PET/SD_dataests/unconditional/samples/00549399/2026-03-04-10-40-49/numpy/1000x512x512x1-samples_png'
#output_folder = '/mnt/D/ywy/ALL_PET/pet_brain_out_ddpm_3/unconditional/samples/00549399/2025-09-26-10-21-57/numpy/100x512x512x1-samples_resino_png'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        # 构建完整的 .npy 文件路径
        npy_file_path = os.path.join(input_folder, filename)
        # 读取 .npy 文件
        image_data = np.load(npy_file_path)

        # 如果图像数据是单通道且维度为 (H, W, 1)，则去掉最后一个维度
        if image_data.ndim == 3 and image_data.shape[-1] == 1:
            image_data = image_data.squeeze(-1)

        # 归一化图像数据到 [0, 255] 范围
        image_data = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)

        # 创建 PIL 图像对象
        image = Image.fromarray(image_data)

        # 构建输出 .png 文件路径
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_png_path = os.path.join(output_folder, output_filename)

        # 保存为 .png 图像
        image.save(output_png_path)

        print(f'已将 {npy_file_path} 保存为 {output_png_path}')