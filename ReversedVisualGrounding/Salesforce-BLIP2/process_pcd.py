import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import sys

# Ensure the PhraseCutDataset path is accessible
sys.path.append("./PhraseCutDataset")

# Import the dataset loader from the provided directory
from utils.refvg_loader import RefVGLoader

# Adjust image path relative to the current working directory
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

# Initialize dataset loaders for train and validation splits
train_loader = RefVGLoader(split='train')
val_loader = RefVGLoader(split='val')

# Function to process and save data as .pth files
def process_and_save(loader, split_name):
    images, masks, captions = [], [], []

    # Loop over all image IDs available in the loader
    for img_id in tqdm(loader.img_ids):
        img_ref_data = loader.get_img_ref_data(img_id=img_id)
        height, width = img_ref_data['height'], img_ref_data['width']
        image_path = os.path.join(IMAGE_ROOT, f"{img_id}.jpg")
        
        if(not os.path.exists(image_path)):
            continue

        # Load the image and resize to 224x224
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert to (C, H, W)

        for task_i, task_id in enumerate(img_ref_data['task_ids']):
            # Get phrase, polygon, and convert polygon to binary mask
            phrase = img_ref_data['phrases'][task_i]
            polygon = img_ref_data['gt_Polygons'][task_i]
            polygon = polygon[0][0]
            mask = polygon_to_mask(polygon, height, width)
            mask = Image.fromarray(mask).resize((224, 224))  # Resize mask to 224x224
            mask_tensor = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)  # (1, H, W)

            # Store the image, mask, and caption
            images.append(image_tensor)
            masks.append(mask_tensor)
            captions.append(phrase)

    # Save the processed data as .pth files
    torch.save(images, f"{split_name}_images.pth")
    torch.save(masks, f"{split_name}_masks.pth")
    torch.save(captions, f"{split_name}_captions.pth")

# Process and save train and validation splits
process_and_save(train_loader, "train")
process_and_save(val_loader, "val")
