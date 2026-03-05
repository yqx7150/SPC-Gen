import os

# 图片所在目录
image_dir = '/mnt/D/ywy/ALL_PET/Dual/chest_transverse/val/18F_sino_png'
# 保存文件名的文本文件路径
txt_file_path = '/mnt/D/ywy/ALL_PET/Dual/chest_transverse/val.txt'

# 获取图片目录下的所有文件
file_names = os.listdir(image_dir)

# 筛选出图片文件（这里假设图片文件扩展名为.jpg）
image_names = [file_name for file_name in file_names if file_name.endswith('.png')]

# 打开文本文件以写入模式
with open(txt_file_path, 'w') as txt_file:
    # 遍历图片文件名列表
    for image_name in image_names:
        # 将文件名写入文本文件，并添加换行符
        txt_file.write(image_name + '\n')

print(f"图片文件名已成功保存到 {txt_file_path}")
    