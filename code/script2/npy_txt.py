import os

# 文件所在目录
npy_dir = '/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_chest_transverse/val/18F_sino'
# 保存文件名的文本文件路径
txt_file_path = '/mnt/D/ywy/ALL_PET/ywy_dataests/Dual_chest_transverse/val.txt'

# 获取目录下的所有文件
file_names = os.listdir(npy_dir)

# 筛选出npy文件
npy_names = [file_name for file_name in file_names if file_name.endswith('.npy')]

# 打开文本文件以写入模式
with open(txt_file_path, 'w') as txt_file:
    # 遍历npy文件名列表
    for npy_name in npy_names:
        # 将文件名写入文本文件，并添加换行符
        txt_file.write(npy_name + '\n')

print(f"npy文件名已成功保存到 {txt_file_path}")