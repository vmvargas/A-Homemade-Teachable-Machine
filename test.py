# import the necessary packages
import CONST
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import time
from keras import applications
from gtts import gTTS
from pygame import mixer
import shutil

#load previously trained model
model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(CONST.IMAGE_SIZE,CONST.IMAGE_SIZE,3)) 
top_model = load_model(os.path.join(CONST.SAVE_DIR,CONST.BOTTLENECK_MODEL))

#load labels
labels = os.listdir('./dataset')

#store the sound of each label
if os.path.exists("./sounds") == True:
    shutil.rmtree("./sounds")
    
os.makedirs("./sounds")
(gTTS(text=labels[0], lang='en')).save("./sounds/0.mp3")
(gTTS(text=labels[1], lang='en')).save("./sounds/1.mp3")
(gTTS(text=labels[2], lang='en')).save("./sounds/2.mp3")
mixer.init()

# Turn on the webcam
cap = cv2.VideoCapture(0)
time.sleep(2)

print ('Press q to exit')
y_pred_old = '-1'
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #preprocessing frame to predict its label
    frame2 = cv2.resize(frame, (CONST.IMAGE_SIZE, CONST.IMAGE_SIZE))
    frame2 = img_to_array(frame2)
    frame2 = np.array(frame2, dtype="float32") / 255
    # generating a prdiction of the frame  
    #y_pred = model.predict_classes(frame2[None,:,:,:])
    
    y_pred = top_model.predict_classes(model.predict(frame2[None,:,:,:]))
    
    if(y_pred[0] != y_pred_old): 
        mixer.music.load("./sounds/"+str(y_pred[0])+'.mp3')
        mixer.music.play()
    
    y_pred_old = y_pred[0]
    
    cv2.putText(frame, labels[y_pred[0]] , (10, 30), CONST.FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press q to exit", (10, 450), CONST.FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()
#remove sounds folder
