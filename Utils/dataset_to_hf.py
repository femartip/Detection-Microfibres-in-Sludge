import os
import json
from datasets import Dataset, Features, Image, Sequence, Value

def load_image_and_mask(image_path, mask_path):
    with open(image_path, "rb") as img_file:
        image = img_file.read()

    mask_data = {"polygons": []}
    
    with open(mask_path, "r") as mask_file:
        try:
            mask_json = json.load(mask_file)
            for polygon in mask_json:
                points = [[point["x"], point["y"]] for point in polygon["content"]]
                label = polygon["labels"]["labelName"]
                mask_data["polygons"].append({"points": points, "label": label})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading mask {mask_path}: {e}")

    return {"image": image, "mask": mask_data}

def create_dataset(images_dir, masks_dir):
    data = []
    for image_filename in os.listdir(images_dir):
        if image_filename.endswith(".jpg"):
            image_path = os.path.join(images_dir, image_filename)
            mask_filename = os.path.splitext(image_filename)[0] + ".json"
            mask_path = os.path.join(masks_dir, mask_filename)

            if os.path.exists(mask_path):
                data.append(load_image_and_mask(image_path, mask_path))

    features = Features({
        "image": Image(),
        "mask": {
            "polygons": Sequence({
                "points": Sequence(Sequence(Value("float32"))),
                "label": Value("string")
            })
        }
    })
    return Dataset.from_list(data, features=features)

if __name__ == '__main__':
    dataset_ca_path = "./data/Dataset_CA/images"
    dataset_glass_path = "./data/Dataset_vidrio/images"
    masks_ca_path = "./data/Dataset_CA/masks"
    masks_glass_path = "./data/Dataset_vidrio/masks"

    dataset_ca = create_dataset(dataset_ca_path, masks_ca_path)
    dataset_glass = create_dataset(dataset_glass_path, masks_glass_path)
    
    print("Dataset CA:")
    print(dataset_ca)
    dataset_ca.push_to_hub("femartip/microfibres_CA_filter")
    print("Dataset saved to HuggingFace")

    print("Dataset Glass:")
    print(dataset_glass)
    dataset_glass.push_to_hub("femartip/microfibres_Glass_filter")
    print("Dataset saved to HuggingFace")