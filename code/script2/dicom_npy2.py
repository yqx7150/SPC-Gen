import pydicom
import numpy as np
import os
from skimage import transform

# 加载 DICOM 文件
def load_dcm(dicom_file):
    return pydicom.dcmread(dicom_file)

# 计算 SUV
def calculate_suv(ds):
    # 提取图像数据并进行缩放
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        image_data = (ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept).astype(np.float32)
    else:
        image_data = ds.pixel_array.astype(np.float32)

    # 体重 (kg)
    patient_weight = ds.PatientWeight  # 单位: kg
    if patient_weight is None or patient_weight == 0:
        raise ValueError("Patient weight is missing or zero.")

    # 注射的放射性剂量 (MBq)
    if 'RadiopharmaceuticalInformationSequence' in ds:
        radiopharmaceutical_info = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose_bq = radiopharmaceutical_info.RadionuclideTotalDose  # 单位: Bq
        if injected_dose_bq is None or injected_dose_bq == 0:
            raise ValueError("Injected dose is missing or zero.")
        injected_dose_mbq = injected_dose_bq / 1e6  # 转换为 MBq
    else:
        raise ValueError("RadiopharmaceuticalInformationSequence not found")

    # 计算 SUV
    suv = (image_data * patient_weight * 1000) / injected_dose_mbq

    # 打印 SUV 的最小值和最大值，用于调试
    print(f"SUV Min: {np.min(suv):.2f}, SUV Max: {np.max(suv):.2f}")

    return suv

# 调整图像大小到 512x512
def resize_image(image, target_shape=(512, 512)):
    return transform.resize(image, target_shape, mode='constant', anti_aliasing=True)

# 保存 numpy 数组
def save_numpy_array(array, output_filename):
    np.save(output_filename, array)
    print(f"Array saved to {output_filename}")

# 主程序
if __name__ == '__main__':
    # DICOM 文件夹路径（修改为指定文件夹）
    dicom_folder_path = "/mnt/D/ywy/ALL_PET/Trunk_pet_dataest/Trunk_pet_dataest_dicom"
 
    # 生成输出目录（修改为指定文件夹）
    output_directory = "/mnt/D/ywy/ALL_PET/Trunk_pet_dataest/Trunk_npy"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
 
    # 获取文件夹中所有 DICOM 文件（包括子文件夹中的文件）
    dicom_file_paths = []
    for root, _, files in os.walk(dicom_folder_path):
        for f in files:
            if f.endswith('.dcm'):
                dicom_file_paths.append(os.path.join(root, f))

    # 处理每个 DICOM 文件
    for idx, dicom_file_path in enumerate(dicom_file_paths):
        # 加载 DICOM 文件
        dcm_data = load_dcm(dicom_file_path)

        # 计算 SUV 并调整大小到 512x512
        try:
            suv_image = calculate_suv(dcm_data)
            suv_image_resized = resize_image(suv_image, (512, 512))

            # 生成唯一的编号文件名
            formatted_idx = f"{idx:05d}"  # 生成 5 位数的索引，如 00000, 00001
            output_filename = os.path.join(output_directory, f"{formatted_idx}.npy")

            # 保存 SUV 图像的 numpy 数组
            save_numpy_array(suv_image_resized, output_filename)
        except ValueError as e:
            print(f"Error processing file {dicom_file_path}: {e}")

    print("Done!")