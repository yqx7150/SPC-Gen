import os




rootPath = "/mnt/D/chenkang/output_512_sin_png/"
imagePath = "/mnt/D/chenkang/output_512_sin_png/val"
file_list = os.listdir(imagePath)
txtName = "val.txt"
with open(rootPath + txtName, "w") as f:
    for i in file_list:
        f.write(i+ "\n")

