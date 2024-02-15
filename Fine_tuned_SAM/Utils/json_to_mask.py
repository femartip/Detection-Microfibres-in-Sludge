import numpy as np
import cv2
import glob
import json

def json_to_mask(json_file, image_shape=(750, 1000)):
    mask = np.zeros(image_shape, dtype=np.uint8)
    #print(mask.shape)
    #print(json_file[0]['content'])
    pts = []
    if len(json_file[0]['content']) == 0:
        return mask
    for point in json_file[0]['content']:
        if json_file[0]['contentType'] == 'polygon':
            #print(point['x'], point['y'])
            x,y = int(point['x']), int(point['y'])
            #print(mask[y,x])
            pts.append([x,y])     
    #print(mask.shape)
    pts = np.array(pts, np.int32)
    mask = cv2.fillPoly(mask, pts=[pts], color=255)    
    return mask

#Read all the masks in json format and save them as an image
 
for file in glob.glob("./data/Dataset/masks/*.json"):
    mask = json_to_mask(json.load(open(file)))
    #print(mask.shape)
    cv2.imwrite("./data/Dataset/ground_truth/{}.jpg".format(file.split("/")[-1].split(".")[0]), mask)