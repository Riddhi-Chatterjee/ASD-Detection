import os
import random
import torch
import numpy as np
from PIL import Image, ImageDraw
import sys

# Ensure the PhraseCutDataset path is accessible
sys.path.append("./PhraseCutDataset")

from utils.refvg_loader import RefVGLoader

# Define the folder to save the samples
OUTPUT_FOLDER = "PCD_Samples"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Path to the images
IMAGE_ROOT = "PhraseCutDataset/data/VGPhraseCut_v0/images"

# Function to convert a polygon to a binary mask
def polygon_to_mask(polygon, height, width):
    """
    Converts a list of polygon points into a binary mask.
    
    Args:
        polygon (list): List of (x, y) coordinates representing the polygon.
        height (int): Height of the image.
        width (int): Width of the image.
    
    Returns:
        mask (np.array): A binary mask with the same size as the image.
    """
    for i in range(len(polygon)):
        polygon[i] = tuple(polygon[i])
    mask = Image.new('L', (width, height), 0)  # Create an empty mask
    ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)  # Draw the polygon
    return np.array(mask, dtype=np.uint8)  # Convert to numpy array

# Initialize the loader
loader = RefVGLoader()

# Get a random image's data
img_ref_data = loader.get_img_ref_data(142)
img_id = img_ref_data['image_id']
height, width = img_ref_data['height'], img_ref_data['width']

# Load the corresponding image
image_path = os.path.join(IMAGE_ROOT, f"{img_id}.jpg")
image = Image.open(image_path).convert('RGB')

# Visualize and save the image, masks, and phrases for 5 tasks
num_tasks = min(5, len(img_ref_data['task_ids']))  # Ensure we don't exceed available tasks

for task_i in range(num_tasks):
    phrase = img_ref_data['phrases'][task_i]
    polygon = img_ref_data['gt_Polygons'][task_i]
    polygon = polygon[0][0]
    mask = polygon_to_mask(polygon, height, width)

    # Save the image
    image.save(os.path.join(OUTPUT_FOLDER, f"{img_id}_image.png"))  # Save image as .png

    # Save the mask as .png
    mask_img = Image.fromarray(mask)
    mask_img.save(os.path.join(OUTPUT_FOLDER, f"{img_id}_mask_{task_i}.png"))

    # Save the text description (phrase) as .txt
    with open(os.path.join(OUTPUT_FOLDER, f"{img_id}_text_{task_i}.txt"), 'w') as f:
        f.write(phrase)

print(f"Saved {num_tasks} tasks with images, masks, and text descriptions to {OUTPUT_FOLDER}.")
