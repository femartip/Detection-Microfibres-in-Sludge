#I have a folder with images which name is the time they where created, the name is as follows 07·45·08.096650_0.png, I would like 
#to rename the images from 930 to 1660, where every time the most recent images with the same second are renamed to the same number.
import os

time_imgs = [img for img in os.listdir("./Fine_tuned_detectron2/data/objectness/") if img.endswith(".png")]
print(len(time_imgs))
time_imgs.sort()
for i in range(930,1660):
    images = time_imgs[0:5]
    time_imgs = time_imgs[5:]
    for j, img in enumerate(images):
        os.rename("./Fine_tuned_detectron2/data/objectness/{}".format(img), "./Fine_tuned_detectron2/data/objectness/{}_{}.png".format(i,j))