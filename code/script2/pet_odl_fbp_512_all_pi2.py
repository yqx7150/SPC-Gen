import numpy as np
import odl
import os
import glob
import time
from tqdm import tqdm

def back_projection(sinogram, geometry, reco_space):
    """使用ODL执行反投影（FBP）算法"""
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    fbp = odl.tomo.fbp_op(ray_trafo)
    projection_element = ray_trafo.range.element(sinogram)
    reconstructed_image_element = fbp(projection_element)
    return reconstructed_image_element.asarray()

def process_batch(input_dir, output_dir, resolution=512, save_png=False, visualize=False):
    """批量处理文件夹中的所有NPY文件并执行反投影
    
    参数:
        input_dir: 输入NPY文件所在文件夹
        output_dir: 输出结果保存文件夹
        resolution: 重建图像分辨率
        save_png: 是否保存PNG格式图像（用于可视化）
        visualize: 是否显示处理过程中的图像
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    if save_png:
        png_output_dir = os.path.join(output_dir, "png")
        os.makedirs(png_output_dir, exist_ok=True)
    
    # 获取所有NPY文件路径
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    if not npy_files:
        print(f"警告: 在 {input_dir} 中未找到NPY文件")
        return
    
    print(f"找到 {len(npy_files)} 个NPY文件，开始批量处理...")
    
    # 定义ODL几何和重建空间
    angle_partition = odl.uniform_partition(0, np.pi, resolution)
    detector_partition = odl.uniform_partition(-360, 360, resolution)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    reco_space = odl.uniform_discr(
        min_pt=[-256, -256], 
        max_pt=[256, 256], 
        shape=[resolution, resolution], 
        dtype='float32'
    )
    
    # 批量处理每个文件
    for file_path in tqdm(npy_files, desc="处理进度"):
        try:
            start_time = time.time()
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            # 读取投影数据
            projection_data = np.load(file_path).astype(np.float32)
            
            # 执行反投影
            reconstructed_image = back_projection(projection_data, geometry, reco_space)
            
            # 保存反投影结果（NPY格式）
            output_npy_path = os.path.join(output_dir, f"{file_name}")
            np.save(output_npy_path, reconstructed_image)
            
            # 可选：保存PNG格式（归一化后）
            if save_png:
                # 归一化以便可视化
                if reconstructed_image.max() > reconstructed_image.min():
                    normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min()) * 255
                else:
                    normalized = np.zeros_like(reconstructed_image)
                normalized = normalized.astype(np.uint8)
                
                # 保存PNG
                from PIL import Image
                output_png_path = os.path.join(png_output_dir, f"{base_name}.png")
                Image.fromarray(normalized).save(output_png_path)
            
            # 可选：可视化
            if visualize:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.imshow(projection_data, cmap='gray')
                plt.title('Projection Data')
                plt.subplot(122)
                plt.imshow(reconstructed_image, cmap='gray')
                plt.title('Reconstructed Image')
                plt.tight_layout()
                plt.show()
            
            # 计算处理时间
            process_time = time.time() - start_time
            print(f"处理完成: {file_name} ({process_time:.2f}秒)")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")

if __name__ == "__main__":
    # 配置参数 - 已修改为指定的文件夹路径
    input_dir = "/mnt/D/ywy/ALL_PET/SD_dataests/unconditional/samples/00549399/2026-03-04-10-40-49/numpy/1000x512x512x1-samples"  # 输入文件夹（投影域文件）
    output_dir = "/mnt/D/ywy/ALL_PET/SD_dataests/unconditional/samples/00549399/2026-03-04-10-40-49/numpy/1000x512x512x1-samples_resino"  # 输出文件夹（反投影域文件）
    #input_dir = "/mnt/D/ywy/ALL_PET/pet_brain_out_ddpm_3/unconditional/samples/00549399/2025-09-26-10-21-57/numpy/100x512x512x1-samples"  # 输入文件夹（投影域文件）
    #output_dir = "/mnt/D/ywy/ALL_PET/pet_brain_out_ddpm_3/unconditional/samples/00549399/2025-09-26-10-21-57/numpy/100x512x512x1-samples_resino"  # 输出文件夹（反投影域文件）
    resolution = 512  # 重建图像分辨率
    save_png = False  # 是否保存PNG格式用于可视化
    visualize = False  # 是否在处理过程中显示图像
    
    # 执行批量处理
    process_batch(input_dir, output_dir, resolution, save_png, visualize)