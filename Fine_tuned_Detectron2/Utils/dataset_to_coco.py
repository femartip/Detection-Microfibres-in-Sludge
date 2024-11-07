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

dataset_name = "vidrio"

image_dir = f'./Fine_tuned_Detectron2/data/Dataset/Dataset_{dataset_name}/images'
mask_dir = f'./Fine_tuned_Detectron2/data/Dataset/Dataset_{dataset_name}/masks'

image_id = 1
annotation_id = 1

coco_format['categories'] = [
    {"id": 1, "name": "dark", "supercategory": "fibre"},
    {"id": 2, "name": "light", "supercategory": "fibre"}
]

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

        # Determine category ID based on folder presence
        if os.path.exists(os.path.join(f'./Fine_tuned_Detectron2/data/Dataset/Dataset_{dataset_name}/light', filename)):
            cat_id = 2
        elif os.path.exists(os.path.join(f'./Fine_tuned_Detectron2/data/Dataset/Dataset_{dataset_name}/dark', filename)):
            cat_id = 1
        else:
            print("Image has no category.")
            continue  # Skip images without a category

        # Open corresponding mask file
        mask_filename = filename.replace('.jpg', '.json')
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"Mask file not found for image {filename}")
            continue

        with open(mask_path) as f:
            mask = json.load(f)

        # Process each object in the mask file
        for obj in mask:
            if len(obj["rectMask"]) == 0:
                continue
            
            x, y, w, h = obj["rectMask"].values()
            bbox = [x, y, w, h]

            # Prepare segmentation points
            seg_mask_points = []
            for mask_points in obj["content"]:
                points = list(mask_points.values())
                x, y = points[0], points[1]
                seg_mask_points.extend([x, y])

            # Append annotation for each object
            coco_format['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": w * h,
                "bbox_mode": 1,  # XYWH_ABS
                "segmentation": [seg_mask_points],
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

# Save the COCO-format JSON
output_path = f'./Fine_tuned_Detectron2/data/Dataset/Dataset_{dataset_name}/coco_format.json'
with open(output_path, 'w') as f:
    json.dump(coco_format, f)

print("COCO format file created successfully at", output_path)
