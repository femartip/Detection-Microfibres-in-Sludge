from PIL import Image
import os

for file in os.listdir("ejemplos_fibras"):
    if file.endswith(".jpg"):
        image = Image.open(os.path.join("ejemplos_fibras", file))
        width, height = image.size
        image = image.resize((int(width/4), int(height/4)))
        image.save(os.path.join("fibras_low_res/", file))
