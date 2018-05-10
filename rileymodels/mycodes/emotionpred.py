from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
# %matplotlib inline
import os

def load_model_dir(model_dir):
    return load_model(os.path.join(model_dir, 'fer2013_mini_XCEPTION.119-0.65.hdf5'))

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def emotionof(model, img):                       # img as array
    # img = cv2.imread(img)                 # image to array
    img = cv2.resize(img,(48,48))         # resize to 48,48
    img = img.mean(axis=2,keepdims=True)  # rgb to grayscale
    img = preprocess_input(img, True)
    x   = np.expand_dims(img,0)           # expand (48,48,1) to (1,48,48,1)
    
    y_pred = model.predict(x)             # predict emotion
    
    labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
        
    label = np.argmax(y_pred)
    confidence = np.amax(y_pred)
    
    return (labels[label],confidence)