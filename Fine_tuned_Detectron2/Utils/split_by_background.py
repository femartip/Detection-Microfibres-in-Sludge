import cv2
import numpy as np
import os

def classify_images(input_dir, output_dir_light, output_dir_dark):
    # Ensure output directories exist
    os.makedirs(output_dir_light, exist_ok=True)
    os.makedirs(output_dir_dark, exist_ok=True)
    light_images = []
    dark_images = []

    for filename in os.listdir(input_dir):
        # Read the image
        img = cv2.imread(os.path.join(input_dir, filename)) # BGR format

        # Calculate the mean brightness
        mean_brightness = np.mean(img, axis=(0, 1))
        #print(filename + str(mean_brightness))
        
        # Classify the image
        if mean_brightness[0] > 150 and mean_brightness[2] < 190:  # You can adjust this threshold as needed
            # The image is 'light'
            output_dir = output_dir_light
            light_images.append(filename)
        else:
            # The image is 'dark'
            output_dir = output_dir_dark
            dark_images.append(filename)

        # Copy the image to the appropriate output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)

    # Print the results
    print(f"Classified {len(light_images)} images as 'light':")
    print(light_images)
    print(f"Classified {len(dark_images)} images as 'dark':")
    print(dark_images)

if __name__ == "__main__":
    # Define the input and output directories
    input_dir = "./Fine_tuned_Detectron2/data/Dataset/images"
    output_dir_light = "./Fine_tuned_Detectron2/data/Dataset/light"
    output_dir_dark = "./Fine_tuned_Detectron2/data/Dataset/dark"

    # Classify the images
    classify_images(input_dir, output_dir_light, output_dir_dark)