import os
import json
from PIL import Image

coco_format = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

coco_format['info'] = {
    "year": "2023",
    "version": "1",
    "description": "Micro Plastic Dataset",
    "contributor": "Felix",
    "url": "",
    "date_created": "12/11/23"
}

image_dir = './Fine_tuned_Detectron2/data/Dataset/images'
mask_dir = './Fine_tuned_Detectron2/data/Dataset/masks'


image_id = 1
annotation_id = 1

coco_format['categories'] = [{"id": 1, "name": "dark", "supercategory": "fibre"},
                             {"id": 2, "name": "light", "supercategory": "fibre"}]

for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        image = Image.open(os.path.join(image_dir, filename))
        width, height = image.size

        # Add image information to 'images' field
        coco_format['images'].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename,
            "license": 1,
            "date_captured": "2023-01-01"
        })
        #If image present in folder ./Fine_tuned_Detectron2/data/Dataset/light
        if os.path.exists(os.path.join('./Fine_tuned_Detectron2/data/Dataset/light', filename)):
            cat_id = 2
        elif os.path.exists(os.path.join('./Fine_tuned_Detectron2/data/Dataset/dark', filename)):
            cat_id = 1
        else:
            print("Image has no category.")

        # Open corresponding mask file
        mask_filename = filename.replace('.jpg', '.json')
        with open(os.path.join(mask_dir, mask_filename)) as f:
            mask = json.load(f)

        if len(mask[0]["rectMask"]) == 0:
            continue

        #print(mask[0]["rectMask"])
        x,y,w,h = mask[0]["rectMask"].values()
        bbox = [x,y,w,h]

        seg_mask_points = []
        for mask_points in mask[0]["content"]:
            points = list(mask_points.values())
            x,y = points[0], points[1]
            seg_mask_points.append(x)
            seg_mask_points.append(y)

        coco_format['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": bbox,
            "area": w*h,
            "bbox_mode": 1,             #XYWH_ABS
            "segmentation": [seg_mask_points],
            "iscrowd": 0
        })

        image_id += 1
        annotation_id += 1

#print(coco_format)

with open('./Fine_tuned_Detectron2/data/coco_format.json', 'w') as f:
    json.dump(coco_format, f)


