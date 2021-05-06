import os
import sys
import numpy as np
from mrcnn import visualize
import cv2
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

#total arguments
n = len(sys.argv)
if n == 2 :
 
    # Arguments passed
 
    print("\nVideo path:", sys.argv[1])
  

    config = Config()
    config.display()
	
	#path to weights file
    MRCNN_model_path = "mrcnn\\mask_rcnn_coco.h5"
	
    print(MRCNN_model_path)
    model = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_model_path, config=Config())
    #load model weights
    model.load_weights(MRCNN_model_path, by_name=True)



    video_name = sys.argv[1]
    cam = cv2.VideoCapture(video_name) 
    fps = cam.get(cv2.CAP_PROP_FPS)
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))


    curr = 0
    #no is used to divide the no frames. if no set to 3, only 1/3rd frame of input video will be there in output
    no = 1
    images = []
  

    while(True): 
	
    	# reading from frame 
        ret,frame = cam.read() 
	
    
        if ret:
            if curr % no ==0:
		
        
	            # Run object detection           
                results1 = model.detect([frame], verbose=1)
            
                r1 = results1[0]
                masked_frame = visualize.display_instances(frame, r1['rois'], r1['masks'], r1['scores'])
                masked_frame = masked_frame.astype(np.uint8)

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
else :
    print("invalid no:of arguements!!")




