import CONST
import numpy as np
import cv2
import os
import time

# helper functions
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def record_frames(video_name):
    # capture a video
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    frame_count = 1
    createFolder('./dataset/'+video_name+'/')
    
    print('Keep r pressed to record')
    print('Keep q pressed to exit')
    
    # record frames
    while(frame_count <= CONST.FRAMES_PER_VIDEO):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Display the resulting frame
        cv2.imshow('frame',frame)
        
        if ret==True:
            if cv2.waitKey(1) & 0xFF == ord('r'):
                print ('.', end="")
                #save the frame as an image
                cv2.imwrite('dataset/%s/%d.jpg' % (video_name, frame_count), frame)     # save frame as JPEG file
                frame_count += 1  
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break  
    # Release opencv resources
    cap.release()
    cv2.destroyAllWindows()

# GETTING THE DATASET 
    
# extract frames from live video
label = input("In a word, what am I about to see?")
record_frames(label)
print ('\nGot it')

label =  input("And now?")
record_frames(label)
print ('\nAlright')

label = input("what about the last?")
record_frames(label)
print ('\nDone')