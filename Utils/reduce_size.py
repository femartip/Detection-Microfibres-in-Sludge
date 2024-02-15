from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
i=0
for file in os.listdir("./Fine_tuned_SAM/data/raw/ejemplos_fibras_all"):
    if file.endswith(".jpg"):
        image = Image.open(os.path.join("./Fine_tuned_SAM/data/raw/ejemplos_fibras_all", file))
        width, height = image.size
        image = image.resize((int(width/4), int(height/4)))
        #image.save(os.path.join("fibras_low_res_all/", str(i)+".jpg"))
        image.save(os.path.join("./Fine_tuned_SAM/data/raw/fibras_low_res_all/", file))
        i+=1
