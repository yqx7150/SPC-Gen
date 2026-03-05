import numpy as np
import os

# 输入文件路径
npz_path = "/mnt/D/ywy/ALL_PET/SD_dataests/unconditional/samples/00549399/2026-03-04-10-40-49/numpy/1000x512x512x1-samples.npz"

# 输出目录
output_dir = '/mnt/D/ywy/ALL_PET/SD_dataests/unconditional/samples/00549399/2026-03-04-10-40-49/numpy/1000x512x512x1-samples'
os.makedirs(output_dir, exist_ok=True)

# 加载 .npz 文件
with np.load(npz_path) as data:
    array = data["images"]
    # 使用数组实际长度作为循环范围
    for idx in range(array.shape[0]):  # 关键修改：用array.shape[0]获取第一个维度的长度
        single_image = array[idx]
        output_path = os.path.join(output_dir, f"{idx}.npy")
        np.save(output_path, single_image)
        print(f"已保存 {output_path}")

print(f"拆分完成！共生成 {array.shape[0]} 个 .npy 文件。")  # 这里也同步修改