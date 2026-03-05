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

# 进行图像旋转
def rotate_image(image, angle):
    pil_image = Image.fromarray(image)
    rotated_image = pil_image.rotate(angle, resample=Image.BILINEAR)
    return np.array(rotated_image)

# 采用Radon变换的正投影过程
def forward_projection(Lambda, theta):
    m, n = Lambda.shape
    # 根据投影线的数量，初始化空白的正弦图
    sinogram = np.zeros((len(theta), n), dtype=np.float32)
    for i, angle in enumerate(theta):
        # 当前图像旋转指定角度，表示不同角度的探测器投影线
        rotated = rotate_image(Lambda, angle)
        # 对一条投影线上的值求和，写入空白的正弦图
        sinogram[i, :] = rotated.sum(axis=0)

    return sinogram

class CT_Sin_masked_datasets(Dataset):
    def __init__(self,
                 root,
                 output_dir,  # 新增输出目录参数
                 mask,
                 need_suv=False,
                 need_norm=False,
                 norm_type="MaxValue",  # MaxValue: 最大值归一化，MaxMin: 最大最小值归一化
                 resolution=512,  # 保持512分辨率
                 ):
        super().__init__()
        self.dataPaths = glob.glob(f"{root}/*.npy")
        self.output_dir = output_dir  # 保存输出目录
        self.mask = mask
        self.resolution = resolution
        self.trans = T.Compose([
            T.ToTensor(),
            # 保持原有变换配置
        ])

        # 保持ODL几何和射线变换配置
        angle_partition = odl.uniform_partition(0, np.pi, resolution)
        detector_partition = odl.uniform_partition(-360, 360, resolution)
        self.geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        self.reco_space = odl.uniform_discr(min_pt=[-256, -256], max_pt=[256, 256], shape=[512, 512], dtype='float32')
        self.ray_trafo = odl.tomo.RayTransform(self.reco_space, self.geometry)
        self.fbp = odl.tomo.fbp_op(self.ray_trafo)

    def __getitem__(self, index):
        # 加载npy文件数据
        pixel_map = np.load(self.dataPaths[index])
        image_element = self.reco_space.element(pixel_map.astype(np.float32))

        # 执行正投影
        projection_data = self.ray_trafo(image_element)
        sinogram = projection_data.asarray()

        # 保存投影域文件到指定输出目录（使用原文件名）
        # 获取原文件名称（不含路径）
        filename = os.path.basename(self.dataPaths[index])
        # 构建保存路径
        save_path = os.path.join(self.output_dir, filename)
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        # 保存npy文件
        np.save(save_path, sinogram.astype(np.float32))

        # 保持后续处理逻辑不变
        sinogram = self.trans(sinogram)

        # 反投影（FBP重建）
        projection_element = self.ray_trafo.range.element(projection_data)
        fbp_op = odl.tomo.fbp_op(self.ray_trafo)
        reconstructed_image_element = fbp_op(projection_element)
        reconstructed_image = reconstructed_image_element.asarray()
        reconstructed = reconstructed_image

        # 可视化重建图像（如需关闭可视化可注释掉以下代码）
        plt.figure(figsize=(8, 8))
        plt.imshow(reconstructed, cmap='gray')
        plt.title('Reconstructed Image (FBP)')
        plt.colorbar()
        plt.axis('off')
        plt.show()

        return {
            "lq_image": sinogram,
            "mask": self.mask,
            "gt_image": sinogram,
        }

    def __len__(self):
        return len(self.dataPaths)

if __name__ == "__main__":
    # 指定输入和输出文件夹路径
    input_root = "/mnt/D/ywy/ALL_PET/CT_dataests/lung/lung_dcm_npy_0_2000"  # 输入npy文件所在目录
    output_dir = "/mnt/D/ywy/ALL_PET/CT_dataests/lung/lung_dcm_npy_0_2000_sino"  # 投影域文件输出目录
    mask = np.ones((512, 512))  # 保持掩码配置
    
    # 初始化数据集并处理
    dataset = CT_Sin_masked_datasets(input_root, output_dir, mask)
    # 遍历所有文件进行处理
    for i in range(len(dataset)):
        dataset[i]