"""
Face detection
"""
import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
import pylab
from utils import load_image
import dlib
import copy
from utils import display_cv2_image

import rileymodels.mycodes.emotionpred as emotion
import rileymodels.mycodes.genderpred as gender



class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label_top(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    @classmethod
    def draw_label_bottom(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y), (x + size[0], y + size[1]), (255, 0, 0), cv2.FILLED)
        point = x, y+size[1]
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def get_expanded_face(self, img, bb):
        img_h, img_w, _ = np.shape(img)
        x1, y1, x2, y2, w, h = bb.left(), bb.top(), bb.right() + 1, bb.bottom() + 1, bb.width(), bb.height()
        xw1 = max(int(x1 - 0.4 * w), 0)
        yw1 = max(int(y1 - 0.4 * h), 0)
        xw2 = min(int(x2 + 0.4 * w), img_w - 1)
        yw2 = min(int(y2 + 0.4 * h), img_h - 1)
        return cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.face_size, self.face_size))

    def detect_face(self, img, emotion_model, gender_model):
        # for face detection
        detector = dlib.get_frontal_face_detector()
            
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        face_bbs = detector(input_img, 1)
        face_imgs = np.empty((len(face_bbs), self.face_size, self.face_size, 3))

        if len(face_bbs) > 0:
            for i, bb in enumerate(face_bbs):
                x1, y1, x2, y2, w, h = bb.left(), bb.top(), bb.right() + 1, bb.bottom() + 1, bb.width(), bb.height()
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                face_imgs[i, :, :, :] = self.get_expanded_face(img, bb)
            
        if len(face_imgs) > 0:
            # predict ages and genders of the detected faces
            results = self.model.predict(face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            
        # draw results
        for i, bb in enumerate(face_bbs):
            # Display age and gender at top of face
            label = "{}, {}".format(int(predicted_ages[i]),
                                    "F" if predicted_genders[i][0] > 0.5 else "M")
            self.draw_label_top(img, (bb.left(), bb.top()), label)
            
            # Display emotion and gender at bottom of face
            emotion_result = emotion.emotionof(emotion_model, face_imgs[i])[0]
            gender_result = gender.genderof(gender_model, face_imgs[i])[0]
            label = "{}, {}".format(emotion_result, gender_result)
            self.draw_label_bottom(img, (bb.left(), bb.bottom()), label)
        return img


# Load riley models
emotion_model = emotion.load_model_dir("rileymodels/trained_models")
gender_model = gender.load_model_dir("rileymodels/trained_models")

# Load sample image
img = load_image("unknown/unknown5.jpg")
# Convert cv2 RBG back to RGB format
#img = img[:,:,::-1]        

face = FaceCV()
img2 = copy.deepcopy(img)
image = face.detect_face(img2, emotion_model, gender_model)


DISPLAY_CV_IMAGE=False

if DISPLAY_CV_IMAGE == True:
    display_cv2_image(image, is_rgb=True)
else:
    pylab.imshow(image)

