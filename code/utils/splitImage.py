from PIL import Image
import os


# 分割训练中的sample_image
from PIL import Image

from PIL import Image

def split_stitched_images(input_path, output_path, image_name, size, padding):
    try:
        # 打开拼接的图片
        image = Image.open(input_path)

        # 获取图片的宽度和高度
        width, height = image.size
        # 检查图片是否是由两个带有 padding 的指定尺寸的图片横向拼接而成
        expected_width = 2 * (size + padding + 1)
        expected_height = size + 2 * padding

        if width == expected_width and height == expected_height:
            # 分割图片，去除 padding
            left_image = image.crop((padding, padding, size + padding, size + padding))
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            left_image.save(os.path.join(output_path, "left_" + image_name))
            print("图片分割成功！")
        else:
            print("输入的图片尺寸不符合要求，请确保是两个带有指定 padding 的指定尺寸的图片横向拼接。")
    except FileNotFoundError as e:
        print(e)
        print("错误：未找到指定的图片文件。")
    except Exception as e:
        print(f"发生未知错误：{e}")



def split_mult(rootPath, outpath, number = 10, prefix="sample"):
    images = os.listdir(rootPath)
    sample_images = list(filter(lambda x : x.startswith(prefix), images))
    for i in range(min(number, len(sample_images))):
        image_path = os.path.join(rootPath, sample_images[i])
        split_stitched_images(image_path, outpath, sample_images[i], 512, 2)



if __name__ == "__main__":
    rootPath = "/mnt/D/yxb/stable-diffusion-master/logs/2025-04-03T14-17-03_CT_PET_512_L_VQ/images/train"
    outPath = os.path.join(rootPath, "split")
    split_mult(rootPath, outPath)