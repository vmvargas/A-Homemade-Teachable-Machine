import CONST
import numpy as np
import cv2
import os
import time
from imutils import paths
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from PIL import Image

EPOCHS = 50
BATCH_SIZE = 10
NUM_CLASSES = 3
TEST_SIZE = 0.25
TRAIN_SAMPLES = CONST.FRAMES_PER_VIDEO*NUM_CLASSES*(1-TEST_SIZE) #112
VAL_SAMPLES = CONST.FRAMES_PER_VIDEO*NUM_CLASSES*TEST_SIZE #38

data = []
labels = []

# helper function to plot a history of model's accuracy and loss
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
# PREPROCESSING DATASET -----------------
print("[INFO] loading dataset...")

#grab the paths to our input images followed by shuffling them 
imagePaths = sorted(list(paths.list_images('dataset')))

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, resize it to IMAGE_SIZE, store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (CONST.IMAGE_SIZE, CONST.IMAGE_SIZE))
    image = img_to_array(image)
    data.append(image)
    
    # extract the class label from the image path and add it to the labels list
    label = imagePath.split(os.path.sep)[-2]
    if label == os.listdir('./dataset')[0]:
        label = 0
    elif label == os.listdir('./dataset')[1]:
        label = 1
    elif label == os.listdir('./dataset')[2]:
        label = 2
    labels.append(label)

# scaling the data points from [0, 255] to the range [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=TEST_SIZE, random_state=CONST.RANDOM_SEED, stratify=labels)

## to make sure  images look correct
#Image.fromarray((X_train[-1]* 255).round().astype(np.uint8))

#from collections import Counter
#Counter(y_train)
#Counter(y_test)

# convert the labels from integers to vectors
y_train = to_categorical(y_train, NUM_CLASSES).astype(int)
y_test = to_categorical(y_test, NUM_CLASSES).astype(int)

## TRAINING THE MODEL -----------------

def save_bottlebeck_features():
    
    print("[INFO] training the VGG16 bottleneck features ...")
    
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=X_train[0].shape) 
    
    bottleneck_features_train = model.predict(X_train, verbose=1)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    
    bottleneck_features_validation  = model.predict(X_test, verbose=1)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


print("[INFO] training the top of the model...")

#save_bottlebeck_features()

train_data = np.load('bottleneck_features_train.npy')
validation_data = np.load('bottleneck_features_validation.npy')

#initialise top model
top_model = Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.6))
top_model.add(Dense(NUM_CLASSES, activation='sigmoid'))

top_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

start = time.time()
model_info = top_model.fit(
        train_data, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_data, y_test),
        shuffle=True,
        verbose=3)
end = time.time()

print ("\nModel training time: %0.1fs\n" % (end - start))

plot_model_history(model_info)

# Evaluating the trained model
scores = top_model.evaluate(validation_data, y_test)
print("\nTest Loss:  %.2f%%" % (scores[0]*100))
print("Test Accuracy: %.2f%%\n" % (scores[1]*100))

# Saving model
if not os.path.isdir(CONST.SAVE_DIR):
    os.makedirs(CONST.SAVE_DIR)
model_path = os.path.join(CONST.SAVE_DIR, CONST.BOTTLENECK_MODEL)
top_model.save(model_path)
print('\nSaved trained model at %s ' % model_path)