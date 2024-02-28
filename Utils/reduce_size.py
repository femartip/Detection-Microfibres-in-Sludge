from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
i=0
for file in os.listdir("./data/raw/fibras_ca_all"):
    if file.endswith(".jpg"):
        image = Image.open(os.path.join("./data/raw/fibras_ca_all", file))
        width, height = image.size
        image = image.resize((int(width/3), int(height/3)))
        #image.save(os.path.join("fibras_low_res_all/", str(i)+".jpg"))
        image.save(os.path.join("./data/raw/fibras_ca_low_res_all/", file))
        i+=1
