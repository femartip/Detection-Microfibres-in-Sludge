# Dataset/masks contains all the masks for the images as json files. Its structure is as follows:
# [{"content": [], "rectMask": {}, "labels": {}, "labelLocation": {}, "contentType": "None"}]
# They are ordered form image 0 to image 1660, but there are some missing masks. This script fills the missing masks with
# empty masks, so the dataset is complete.
import os
import json

masks = [int(mask.split(".")[0]) for mask in os.listdir("./Fine_tuned_Detectron2/data/Dataset/masks/") if mask.endswith(".json")]
#print(masks)
masks.sort()

for i in range(0,1660):
    if masks[i] != i:
        masks.insert(i,i)
        with open("./Fine_tuned_Detectron2/data/Dataset/masks/{}.json".format(i), "w") as f:
            json.dump([{"content": [], "rectMask": {}, "labels": {}, "labelLocation": {}, "contentType": "None"}], f)
        print("Added mask {}".format(i))
    else:
        print("Mask {} is ok".format(i))