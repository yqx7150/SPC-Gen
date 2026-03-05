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
            gray_image[gray_image < 20] = 0
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


# 采用 ODL 进行反投影
def back_projection(sinogram, geometry, reco_space):
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    fbp = odl.tomo.fbp_op(ray_trafo)
    projection_element = ray_trafo.range.element(sinogram)
    reconstructed_image_element = fbp(projection_element)
    return reconstructed_image_element.asarray()

if __name__ == "__main__":
    root = "/mnt/D/yxb/stable-diffusion-master/out/log/unconditional/samples/00273443/2025-04-02-18-48-25/numpy"
    input_npz_path = '100x256x256x3-samples.npz'
    output_npz_path = '10x256x256x3-out-samples.npz'
    n = 10  # 要补齐的列数
    pad_images_with_average(os.path.join(root, input_npz_path), os.path.join(root, output_npz_path), n)
    # 处理之后的数据
    out_images = np.load(os.path.join(root, output_npz_path))['images']
    # 读取投影域图片
    for i,projection_image in enumerate(out_images):

        # 将图像重塑到 540x540
        # projection_image = projection_image.resize((540, 540), Image.BILINEAR)
        projection_image = np.array(projection_image).astype(np.float32)
        # 定义 ODL 所需的几何和重建空间
        resolution = 256  # 这里也需要根据新的尺寸调整
        angle_partition = odl.uniform_partition(0, 2 * np.pi, resolution)
        detector_partition = odl.uniform_partition(-360, 360, resolution)
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        reco_space = odl.uniform_discr(min_pt=[-256, -256], max_pt=[256, 256], shape=[512, 512], dtype='float32')  # 根据新尺寸调整

        # 进行反投影
        #projection_image=projection_image[:,:,0]
        reconstructed_image = back_projection(projection_image, geometry, reco_space)

        # 归一化重建图像以便保存
        reconstructed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min()) * 255
        reconstructed_image_normalized = reconstructed_image_normalized.astype(np.uint8)

        # 保存反投影后的图片
        output_dir = os.path.join(root, "last_image") # 输出目录也相应修改
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{i}.png")
        reconstructed_image_pil = Image.fromarray(reconstructed_image_normalized)
        reconstructed_image_pil.save(output_path)



