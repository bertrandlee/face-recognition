from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
# %matplotlib inline

model = load_model('trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5')

def emotionof(img):                       # img as array
    # img = cv2.imread(img)                 # image to array
    img = cv2.resize(img,(48,48))         # resize to 48,48
    img = img.mean(axis=2,keepdims=True)  # rgb to grayscale
    x   = np.expand_dims(img,0)           # expand (48,48,1) to (1,48,48,1)
    
    y_pred = model.predict(x)             # predict emotion
    
    labels = {0:'angry',1:'disgust',2:'sad',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    
    label = np.argmax(y_pred)
    confidence = np.amax(y_pred)
    
    return (labels[label],confidence)