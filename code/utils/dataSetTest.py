from torch.utils.data import DataLoader
from ldm.data.PetData import PET_SIN_512_train,PET_SIN_512_val
from PIL import Image

petdata = PET_SIN_512_train()
image = petdata[0]['image']
print(image[:,1,:])
image.save("./aaa.png")
print()