# 将图像重塑到 540x540
# projection_image = projection_image.resize((540, 540), Image.BILINEAR)
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
root = "./"
def back_projection(sinogram, geometry, reco_space):
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    fbp = odl.tomo.fbp_op(ray_trafo)
    projection_element = ray_trafo.range.element(sinogram)
    reconstructed_image_element = fbp(projection_element)
    return reconstructed_image_element.asarray()
projection_image = Image.open("582.jpg")
projection_image = np.array(projection_image).astype(np.float32)
# 定义 ODL 所需的几何和重建空间
resolution = 512 # 这里也需要根据新的尺寸调整
angle_partition = odl.uniform_partition(0, 2 * np.pi, resolution)
detector_partition = odl.uniform_partition(-360, 360, resolution)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
reco_space = odl.uniform_discr(min_pt=[-256, -256], max_pt=[256, 256], shape=[512, 512], dtype='float32')  # 根据新尺寸调整

# 进行反投影
# projection_image=projection_image[:,:,0]
reconstructed_image = back_projection(projection_image, geometry, reco_space)

# 归一化重建图像以便保存
reconstructed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (
            reconstructed_image.max() - reconstructed_image.min()) * 255
reconstructed_image_normalized = reconstructed_image_normalized.astype(np.uint8)

# 保存反投影后的图片
output_dir = os.path.join(root, "last_image")  # 输出目录也相应修改
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{0}.png")
reconstructed_image_pil = Image.fromarray(reconstructed_image_normalized)
reconstructed_image_pil.save(output_path)