#Given the images in data/Dataset/images, if an image does not have a corresponding mask, then create a mask of all zeros
import os 
import cv2
import json
import numpy as np

for file in os.listdir("../data/Dataset/images"):
    if file.endswith(".jpg"):
        filename = file.split(".")[0]
        image = cv2.imread(os.path.join("../data/Dataset/images/", file))
        x,y = image.shape[:2]
        if not os.path.exists(os.path.join("../data/Dataset/masks/", filename + ".json")):
            print("Creating empty mask for image {}".format(file))
            #The json file should only contain an list, that contains a dictionary with the key "content" and the value being an empty list
            json_file = [{"content": [], "rectMask": {}, "labels": {}, "labelLocation": {}, "contentType": "None"}]
            with open(os.path.join("../data/Dataset/masks/", filename + ".json"), "w") as f:
                json.dump(json_file, f)
