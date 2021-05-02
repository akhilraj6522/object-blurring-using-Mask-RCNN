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

# total arguments
n = len(sys.argv)
if n == 2 :
 
    # Arguments passed
 
    print("\nImage path:", sys.argv[1])
    

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



    image = cv2.imread(sys.argv[1])
    if image is not None:
            

        # Run object detection to extract dimension from a single image
        print(len([image]))
        results1 = model.detect([image], verbose=1)
       
        r1 = results1[0]
        img = visualize.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],r1['scores'], ax=None)
        img = img.astype(np.uint8).copy()
        cv2.imshow('jj',img)
        cv2.imwrite('out.jpg',img)
        cv2.waitKey(0)
else :
    print("invalid no:of arguements!!")




