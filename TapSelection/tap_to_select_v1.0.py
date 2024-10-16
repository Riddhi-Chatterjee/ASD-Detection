import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch

mouseX = 0
mouseY = 0

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      global mouseX
      global mouseY
      mouseX = x
      mouseY = y
      

def get_selected_masks(idx_pt, masks): #Returns a list of masks
    selected_masks = []
    for mask in masks:
        if mask[idx_pt[0]][idx_pt[1]] != 0:
            selected_masks.append(mask)
    return selected_masks


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
video_path = "./objects-fixed-cam.mp4"
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
        results = model(frame)
        
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
        selected_masks = get_selected_masks((mouseY, mouseX), resized_masks)
        
        mask_union = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
        if(len(selected_masks) != 0):
            mask_union = np.clip(sum(selected_masks), a_min = 0, a_max = 1) 
            mask_union = (mask_union * 255).astype(np.uint8)
            cv2.imshow("Selected Masks", mask_union)
        else:
            print("No object is selected!")
        
        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # # # Display the annotated frame
        # cv2.putText(img = annotated_frame, text = f"FPS: {int(fps)}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (0, 0, 255), thickness = 4, lineType = cv2.LINE_AA)
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        seg_frame = highlight_mask(frame, mask_union)
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
        