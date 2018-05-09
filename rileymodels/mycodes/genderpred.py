
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
# %matplotlib inline
import os

def load_model_dir(model_dir):
    return load_model(os.path.join(model_dir, 'gender_models/gender_mini_XCEPTION.21-0.95.hdf5'))

def genderof(model, img):                       # img as array
    # img = cv2.imread(img)                 # image to array
    img = cv2.resize(img,(64,64))         # resize to 64,64
    img = img.mean(axis=2,keepdims=True)  # rgb to grayscale
    x   = np.expand_dims(img,0)           # expand (64,64,1) to (1,64,64,1)
    
    y_pred = model.predict(x)             # predict emotion
    
    labels = {0:'man',1:'woman'}
    
    label = np.argmax(y_pred)
    confidence = np.amax(y_pred)
    
    return (labels[label],confidence)