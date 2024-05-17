# Convert tiff images into patches of 1240x720 and save them as .png images

from PIL import Image
import os
from tqdm import tqdm

def extract_patches(input_path, output_path, patch_size=(256, 256)):
    # Open the image file
    img = Image.open(input_path)
    width, height = img.size

    # Calculate the number of patches in x and y direction
    num_x_patches = width // patch_size[0]
    num_y_patches = height // patch_size[1]

    # Loop through the image and extract patches
    for i in range(num_x_patches):
        for j in range(num_y_patches):
            box = (i*patch_size[0], j*patch_size[1], (i+1)*patch_size[0], (j+1)*patch_size[1])
            patch = img.crop(box)
            
            # Save the patch to the output directory
            patch_filename = f'{os.path.basename(input_path).split(".")[0]}_{i}_{j}.png'
            #Resize image to 1000x750
            patch = patch.resize((1000, 750))
            patch.save(os.path.join(output_path, patch_filename))

    print("Patch extraction completed.")


path = "data/tiff_images/"
#Loop throug tiff images
for file in tqdm(os.listdir(path)):
    mkdir = path + file.split(".")[0] + "/"
    if not os.path.exists(mkdir):
        os.makedirs(mkdir)
    image_path = path + file
    extract_patches(image_path, mkdir)

