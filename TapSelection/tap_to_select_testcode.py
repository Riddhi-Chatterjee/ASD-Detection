# Incorporates object tracking

import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import signal

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


mouseX = None
mouseY = None
new_click = False
selected_id = None

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      global mouseX
      global mouseY
      global new_click
      mouseX = x
      mouseY = y
      new_click = True
      

def get_selected_mask(idx_pt, masks, results): #Returns a list of masks
    global selected_id
    global new_click
    selected_mask = None
    
    if new_click:
        new_click = False
        index = 0
        for mask in masks:
            if mask[idx_pt[0]][idx_pt[1]] != 0:
                selected_mask = mask
                break
            index += 1
        
        if index < len(masks):
            selected_id = results[0].boxes.id[index].item()
        else:
            selected_id = None
    else:
        if selected_id != None:
            index = (results[0].boxes.id == selected_id).nonzero(as_tuple=True)[0]
            if len(index) != 0:
                selected_mask = masks[index[0].item()]
    
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


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Load the YOLOv8 model
model = YOLO('yolov8s-seg.pt')

# Open the video file
video_path = "./media/objects-moving-cam.mp4"
cap = cv2.VideoCapture(video_path)

# create a window
cv2.namedWindow('Segmentation Result')
# bind the callback function to window
cv2.setMouseCallback('Segmentation Result', click_event)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        start = time.perf_counter()

        # Run YOLOv8 inference on the frame
        results = model.track(frame, persist=True)
        
        print(results[0].boxes)
        
        
        end = time.perf_counter()
        total_time = end - start
        fps = 1/total_time
        
        masks = results[0].masks.data
        
        resized_masks = []
        for mask in masks:
            mask = mask.numpy()
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = np.clip(mask, a_min = 0, a_max = 1) 
            resized_masks.append(mask)
        
        print("MouseX: "+str(mouseX))
        print("MouseY: "+str(mouseY))
        selected_mask = get_selected_mask((mouseY, mouseX), resized_masks, results)
        
        ########################################
        if selected_id is not None:
            index = (results[0].boxes.id == selected_id).nonzero(as_tuple=True)[0]
            if len(index) != 0:
                bbox = results[0].boxes.xyxy[index[0].item()].cpu().numpy()
                bbox = bbox.astype(int)
                cropped_object = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cv2.imshow("Cropped Object", cropped_object)
        #########################################
        
        if type(selected_mask) != type(None):
            selected_mask = np.clip(selected_mask, a_min = 0, a_max = 1) 
            selected_mask = (selected_mask * 255).astype(np.uint8)
            cv2.imshow("Selected Mask", selected_mask)
        else:
            selected_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            print("No object is selected!")
        
        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # # # Display the annotated frame
        # cv2.putText(img = annotated_frame, text = f"FPS: {int(fps)}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (0, 0, 255), thickness = 4, lineType = cv2.LINE_AA)
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        seg_frame = highlight_mask(frame, selected_mask)
        cv2.putText(img = seg_frame, text = f"FPS: {int(fps)}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (0, 0, 255), thickness = 4, lineType = cv2.LINE_AA)
        cv2.imshow("Segmentation Result", seg_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(int(total_time*1000)) & 0xFF == ord("q"):
            break
    else:
        #Break the loop if the end of the video is reached
        break
    
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
        