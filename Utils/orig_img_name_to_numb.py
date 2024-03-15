#Read all files from data/raw/fibras_ca_low_res_all and change the name of the file to a number

import os

#file_list = os.listdir("./data/raw/fibras_ca_low_res_all")
file_list = os.listdir("./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/masks/")
file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for i, file in enumerate(file_list):
    image_name = file.split(".")[0] + ".jpg"
    os.rename(os.path.join("./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/images/", image_name), os.path.join("./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/images/", str(i)+".jpg"))
    os.rename(os.path.join("./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/masks/", file), os.path.join("./Fine_tuned_Detectron2/data/Dataset/Dataset_CA/masks/", str(i)+".json"))
    print(file, " -> ", str(i)+".json")


