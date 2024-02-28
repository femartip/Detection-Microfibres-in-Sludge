#Read all files from data/raw/fibras_ca_low_res_all and change the name of the file to a number

import os

file_list = os.listdir("./data/raw/fibras_ca_low_res_all")
file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for i, file in enumerate(file_list):
    os.rename(os.path.join("./data/raw/fibras_ca_low_res_all", file), os.path.join("./data/raw/fibras_ca_low_res_all", str(i)+".jpg"))
    print(file, " -> ", str(i)+".jpg")


