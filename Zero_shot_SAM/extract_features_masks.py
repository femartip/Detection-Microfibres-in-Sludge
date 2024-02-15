#Load the images of masks_h in a list
import os
import cv2
from matplotlib import pyplot as plt

for file in os.listdir("./masks"):
    if file.endswith(".png"):
        image = cv2.imread(os.path.join("./masks/", file))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
        x,y,w,h = cv2.boundingRect(img)
        if h/w > 3 and h > 13 and h < 5000 and not (w >= 196 and w <= 198 and h >= 15 and h <= 18):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img, "w={} micro m,h={} micro m".format(abs((w*197)/750),abs((w*197)/750)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imwrite(os.path.join("./feature_extractor/", file), img)