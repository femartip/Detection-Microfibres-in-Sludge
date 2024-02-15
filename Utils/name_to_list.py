#From ./Fine_tuned_SAM/data/raw/ejemplos_fibras_all save a list of the names of the images

import glob
import os


images = [os.path.basename(file) for file in glob.glob("./Fine_tuned_SAM/data/raw/ejemplos_fibras_all/*.jpg")]
with open("./Utils/ejemplos_fibras_all.txt", "w") as f:
    for image in images:
        f.write(image + ",")
