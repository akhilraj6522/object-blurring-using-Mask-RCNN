"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import cv2
import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


############################################################
#  Visualization
############################################################



def apply_mask(image, mask):
    """Apply the given mask to the image.
    """
    image = image.astype(np.uint8)
    image = np.array(image)
 
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  cv2.blur(image[:, :, c],(40,40)),
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, 
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    print(boxes.shape)
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
	#sorting cordinates of irregular shape for counting objects
    
  
    
    masked_image = image.astype(np.uint32).copy()
	
 
    print(scores.shape)
    
    for i in range(N):
        
        #if scores[i] == max(scores):
        print('score', scores[i], '||||')
        # Mask irregular shape with black and other with random colours
        mask = masks[:, :, i]
        if show_mask:           
            masked_image = apply_mask(masked_image, mask)

            
        
            
            
            
        masked_image = masked_image.astype(np.uint8)


    return masked_image




