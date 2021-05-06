"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import cv2


import numpy as np

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


def display_instances(image, boxes, masks, 
                      scores=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]

    scores: (optional) confidence scores for each box
    
    """
    # Number of instances
    print(boxes.shape)
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    
    
    
    masked_image = image.astype(np.uint32).copy()
	
 
    print(scores.shape)
    
    for i in range(N):
        
        #if scores[i] == max(scores):
        print('score', scores[i], '||||')
 
        mask = masks[:, :, i]
                  
        masked_image = apply_mask(masked_image, mask)
                     
        masked_image = masked_image.astype(np.uint8)


    return masked_image




