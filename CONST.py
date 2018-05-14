# -*- coding: utf-8 -*-
import os
import cv2
# Global variables
RANDOM_SEED = 2017

FRAMES_PER_VIDEO = 50
IMAGE_SIZE = 150

SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
CRAPPY_MODEL = 'crappy_model.h5'
BOTTLENECK_MODEL = 'bottleneck_model.h5'

FONT = cv2.FONT_HERSHEY_SIMPLEX