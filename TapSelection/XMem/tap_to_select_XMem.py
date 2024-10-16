import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import sys
import signal
import os
from os import path
from argparse import ArgumentParser
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import copy
torch.set_grad_enabled(False)

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#################### Installations and downloads #####################
#NOTE: Execute these steps only for the first time this script is run
# os.system("git clone https://github.com/hkchengrex/XMem.git")
# os.system("mv XMem/* .")
# os.system("rm -r XMem")
# os.system("pip install opencv-python")
# os.system("pip install -U numpy")
# os.system("pip install -r requirements.txt")
# # Downloading the pretrained XMem model
# os.system("wget -P ./saves/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth")
######################################################################

############################# Extra imports ##########################
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
######################################################################


mouseX = None
mouseY = None
new_click = False
selected_id = None
num_objects = None
processor = None

device = None
if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'   

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      global mouseX
      global mouseY
      global new_click
      mouseX = x
      mouseY = y
      new_click = True

def get_object_mask(selected_id, predicted_seg_mask):
    object_mask = copy.deepcopy(predicted_seg_mask)
    object_mask[predicted_seg_mask == selected_id] = 1
    object_mask[predicted_seg_mask != selected_id] = 0
    return object_mask
      
def get_selected_mask(idx_pt, resized_object_masks, object_detection_results, frame, init_seg_mask, tracker, config): #Returns the selected mask
    global selected_id
    global new_click
    global num_objects
    global processor
    global device
    selected_mask = None
    
    if type(init_seg_mask) != type(None):
        num_objects = len(np.unique(init_seg_mask)) - 1
        
        torch.cuda.empty_cache()

        processor = InferenceCore(tracker, config=config)
        processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
        
        predicted_seg_mask = None
        with torch.cuda.amp.autocast(enabled=True):
            # convert numpy array to pytorch tensor format
            frame_torch, _ = image_to_torch(frame, device=device)
            # initialize with the init_seg_mask
            mask_torch = index_numpy_to_one_hot_torch(init_seg_mask, num_objects+1).to(device)
            # the background mask is not fed into the model
            prediction = processor.step(frame_torch, mask_torch[1:])
            # argmax, convert to numpy
            predicted_seg_mask = torch_prob_to_numpy_mask(prediction)
        
        selected_id = predicted_seg_mask[idx_pt[0]][idx_pt[1]]
        if selected_id == 0: # Doesn't make sense to select the background
            selected_id = None
        
        if selected_id != None and type(predicted_seg_mask) != type(None):
            selected_mask = get_object_mask(selected_id, predicted_seg_mask)
        #print("Unique elements in init seg mask:", np.unique(init_seg_mask))
    elif processor != None:
        predicted_seg_mask = None
        with torch.cuda.amp.autocast(enabled=True):
            # convert numpy array to pytorch tensor format
            frame_torch, _ = image_to_torch(frame, device=device)
            # propagate only
            prediction = processor.step(frame_torch)
            # argmax, convert to numpy
            predicted_seg_mask = torch_prob_to_numpy_mask(prediction)
            #cv2.imshow("Predicted Seg Mask", predicted_seg_mask.astype(np.uint8))
            print("Selected ID:", selected_id)
            print("Unique elements in predicted seg mask:", np.unique(predicted_seg_mask))
            print("")
        
        if selected_id != None and type(predicted_seg_mask) != type(None):
            selected_mask = get_object_mask(selected_id, predicted_seg_mask)
    
    return selected_mask
        
def highlight_mask(image, mask):
    # Create a copy of the original image
    highlighted_image = image.copy()

    # Set the masked portion to a translucent red color
    highlighted_image[mask != 0] = [0, 0, 255]  # Set the BGR values for red

    # Blend the original image with the highlighted mask using addWeighted
    alpha = 0.5  # Adjust the alpha value for the level of transparency
    final_image = cv2.addWeighted(image, 1 - alpha, highlighted_image, alpha, 0)

    return final_image

def get_initial_segmentation_mask(segmenter, frame):
    # Run YOLOv8 inference on the frame
    results = segmenter(frame)
    
    masks = results[0].masks.data
        
    resized_masks = []
    for mask in masks:
        mask = mask.detach().cpu().numpy()
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask = np.clip(mask, a_min = 0, a_max = 1) 
        resized_masks.append(mask)
    
    segmentation_mask = None
    for i, mask in enumerate(resized_masks):
        if i == 0:
            segmentation_mask = copy.deepcopy(mask)
            segmentation_mask = segmentation_mask.astype(np.int32)
        else:
            filler = i+1
            segmentation_mask[mask == 1] = filler
    
    return segmentation_mask, results, resized_masks
         

# default configuration
XMem_config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

XMem_tracker = XMem(XMem_config, './saves/XMem.pth', map_location=torch.device(device)).eval().to(device)

# Load the YOLOv8 model
yolo_instance_segmenter = YOLO('yolov8s-seg.pt')

# Open the video file
video_path = "../media/objects-taken-away-2.mp4"
cap = cv2.VideoCapture(video_path)
frame_width  = cap.get(3)  # float `width`
frame_height = cap.get(4)  # float `height`

# create a window
cv2.namedWindow('Segmentation Result')
# bind the callback function to window
cv2.setMouseCallback('Segmentation Result', click_event)

#Resize window
resize_dim = (int(frame_width/4), int(frame_height/4))

# Loop through the video frames
frame_count = 0
while cap.isOpened():
    # Read a frame from the video
    start = time.perf_counter()
    success, frame = cap.read()
    
    if success:
        frame = cv2.resize(frame, resize_dim)
        init_seg_mask = None
        object_detection_results = None
        resized_object_masks = None
        if new_click:
            new_click = False
            init_seg_mask, object_detection_results, resized_object_masks = get_initial_segmentation_mask(yolo_instance_segmenter, frame)
        
        print("MouseX: "+str(mouseX))
        print("MouseY: "+str(mouseY))
        
        selected_mask = get_selected_mask((mouseY, mouseX), resized_object_masks, object_detection_results, frame, init_seg_mask, XMem_tracker, XMem_config)
        
        if type(selected_mask) != type(None):
            selected_mask = np.clip(selected_mask, a_min = 0, a_max = 1) 
            selected_mask = (selected_mask * 255).astype(np.uint8)
            selected_mask = cv2.resize(selected_mask, resize_dim)
            cv2.imshow("Selected Mask", selected_mask)
        else:
            selected_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            print("No object is selected!")
        
        seg_frame = highlight_mask(frame, selected_mask)
        end = time.perf_counter()
        total_time = end - start
        fps = 1/total_time
        cv2.putText(img = seg_frame, text = f"FPS: {int(fps)}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (0, 0, 255), thickness = 4, lineType = cv2.LINE_AA)
        seg_frame = cv2.resize(seg_frame, resize_dim)
        cv2.imshow("Segmentation Result", seg_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    else:
        #Break the loop if the end of the video is reached
        break
    frame_count += 1
    
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
        