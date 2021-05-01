import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import visualize

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath(".\\")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib

class Config(Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 81
    DETECTION_MIN_CONFIDENCE = 0.2
    # IMAGE_MIN_DIM = 832
    # IMAGE_MAX_DIM = 832

config = Config()
config.display()
MRCNN_model_path = "mrcnn\\mask_rcnn_coco.h5"
print(MRCNN_model_path)
model = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_model_path, config=Config())
#load model weights
model.load_weights(MRCNN_model_path, by_name=True)
import cv2
import os



video_name = 'cow.mp4'
cam = cv2.VideoCapture(video_name) 
fps = cam.get(cv2.CAP_PROP_FPS)
size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))


curr = 0
#no is used to divide the no frames. if no set to 3, only 1/3rd frame of input video will be there in output
no = 1
images = []
frame_copy = []

while(True): 
	
	# reading from frame 
    ret,frame = cam.read() 
	
    
    if ret:
        if curr % no ==0:
		
        
	        # Run object detection
            frame_copy.append(frame)
            results1 = model.detect([frame], verbose=1)

            
            r1 = results1[0]
            masked_frame = visualize.display_instances(frame, r1['rois'], r1['masks'], r1['class_ids'],
                                        r1['scores'], ax=None,
                                        title="Predictions1")
            masked_frame = masked_frame.astype(np.uint8)
            #cv2.imshow('jj', masked_frame)
            #cv2.waitKey(0)
            images.append (masked_frame)
		   
        curr = curr + 1
       
             
    else: 
        break




out = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(images)):
    # writing to a image array
    out.write(images[i])
out.release()


# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()




