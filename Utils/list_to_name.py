#From the file text ejemplos_fibras_all.txt, make a copy of ./Zero_shot_SAM/fibras_detectadas_h in this folder where the name of the 
#images are changed to the names in the text file

import glob
import os
import shutil

with open("./Utils/ejemplos_fibras_all.txt", "r") as f:
    new_name = f.read().split(",")
    new_name = new_name[:-1]

for i in range(0,1660):
    try:
        shutil.copy("./Zero_shot_SAM/fibras_detectadas_h/{}.jpg".format(i), "./Zero_shot_SAM/fibras_detectadas_h_orig/{}".format(new_name[i]))
    except:
        print(i)
        pass


    
