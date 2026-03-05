import astra
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import transform
from PIL import Image

# 创建几何结构
def set_projection_size(vol_size=(256, 256)):
    global vol_geom, proj_geom, proj_id
    vol_geom = astra.create_vol_geom(vol_size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, vol_size[0], np.linspace(0, np.pi, vol_size[1], False))
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

# 使用 ASTRA 的 ML-EM 算法将正弦图重建为 PET 图像
def sino_to_pet(sinogram, vol_geom, proj_geom, num_iterations=100):
    """
    使用 ASTRA 的 ML-EM 算法将正弦图重建为 PET 图像。
    """
    # 检查并调整正弦图数据的形状
    if sinogram.ndim == 1:
        num_projections = proj_geom['ProjectionAngles'].shape[0]
        num_detectors = proj_geom['DetectorCount']
        sinogram = sinogram.reshape((num_projections, num_detectors))

    # 归一化正弦图数据
    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min() + 1e-10)

    # 创建正弦图 ID
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    # 创建重建图像 ID，并设置非零初始估计
    initial_estimate = np.ones((vol_geom['GridRowCount'], vol_geom['GridColCount']), dtype=np.float32)
    rec_id = astra.data2d.create('-vol', vol_geom, initial_estimate)

    # 创建投影器
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    # 配置 ML-EM 算法
    cfg = astra.astra_dict('EM_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id

    # 创建并运行 ML-EM 算法
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)

    # 获取重建结果
    reconstructed_image = astra.data2d.get(rec_id)

    # 清理资源
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)

    return reconstructed_image

# 保存图像
def save_image(image, output_filename, title=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(image, cmap='gray')
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存

# 调整图像大小到指定尺寸
def resize_image(image, target_shape=(256, 256)):
    return transform.resize(image, target_shape, mode='constant', anti_aliasing=True)

# 保存 numpy 数组
def save_numpy_array(array, output_filename):
    np.save(output_filename, array)
    print(f"Array saved to {output_filename}.npy")

# 主程序
if __name__ == '__main__':
    # 设置投影尺寸
    set_projection_size((256, 256))
    # 加载 256x256 的 npy 数据（这里的数据已经是正弦图）
    image_path = "/mnt/D/yxb/stable-diffusion-master/logs/2025-04-03T14-17-03_CT_PET_512_L_VQ/images/train/split"
    for file_name in os.listdir(image_path):
        if not file_name.endswith("png"):
            continue
        file_path = image_path + file_name
        image = Image.open(file_path)
        image = image.convert("L")
        array = np.array(image)
        # 重建图
        reconstructed_image = sino_to_pet(array, vol_geom, proj_geom, num_iterations=50)

        plt.imsave(image_path + 'out/' + file_name, reconstructed_image, cmap='gray')

    # # # 生成输出目录
    # # output_directory = "/mnt/D/chenkang/high_res_PET"
    # # if not os.path.exists(output_directory):
    # #     os.makedirs(output_directory)
    
    # #  # 保存重建图
    # # reconstructed_image_filename = os.path.join(output_directory, "reconstructed_PET1_256.png")
    # # save_image(reconstructed_image, reconstructed_image_filename, title="Reconstructed Image")
    # # print(f"Reconstructed image saved to {reconstructed_image_filename}")

    # # print("Done!")