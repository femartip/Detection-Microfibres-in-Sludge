#Given a HuggingFace dataset, load it to disk and save the images and masks as images in a folder

import os
import sys
import cv2
import json
import glob
import numpy as np
from datasets import Dataset

dataset = Dataset.load_from_disk("./data/train_929img_3")
images = dataset["image"]
masks = dataset["mask"]

for i, (image, mask) in enumerate(zip(images, masks)):
    image.save("./data/train/images/{}.jpg".format(i))
    mask.save("./data/train/masks/{}.jpg".format(i))
