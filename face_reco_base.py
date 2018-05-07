#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dlib
import cv2
import bz2
import os

from urllib.request import urlopen

# Download dlib face detection landmarks file
def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)

dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)

# Create CNN model and load pretrained weights (OpenFace nn4.small2)
from model import create_model

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

# Load images dataset
import numpy as np
import os.path

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 

ds_store = ".DS_Store"
    
def load_metadata(path):
    metadata = []
    dirs = os.listdir(path)
    if ds_store in dirs:
        dirs.remove(ds_store)
    for i in dirs:
        subdirs = os.listdir(os.path.join(path, i))
        if ds_store in subdirs:
            subdirs.remove(ds_store)
        for f in subdirs:
            metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

metadata = load_metadata('images')

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib
from utils import load_image

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

# Align helper functions
def get_face_thumbnail(img):
    return alignment.getLargestFaceThumbnail(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def get_all_face_thumbnails(img):
    return alignment.getAllFaceThumbnails(96, img, 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Get embedding vectors
embedded = np.zeros((metadata.shape[0], 128))

def get_face_vector(img):
    img = get_face_thumbnail(img)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    return nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

def get_face_vectors(img):
    faces = get_all_face_thumbnails(img)
    face_vectors = []
    for face_img in faces:
        # scale RGB values to interval [0,1]
        face_img = (face_img / 255.).astype(np.float32)
        # obtain embedding vector for image
        vector = nn4_small2_pretrained.predict(np.expand_dims(face_img, axis=0))[0]
        face_vectors.append(vector)
    return face_vectors
    
for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    # obtain embedding vector for image
    embedded[i] = get_face_vector(img)
    
# Train classifier models
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def train_images(metadata, embedded):
    targets = np.array([m.name for m in metadata])

    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 2 != 0
    test_idx = np.arange(metadata.shape[0]) % 2 == 0

    # 50 train examples of 10 identities (5 examples each)
    X_train = embedded[train_idx]
    # 50 test examples of 10 identities (5 examples each)
    X_test = embedded[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()

    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))

    print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
    return encoder, knn, svc, test_idx, targets

encoder,knn,svc,test_idx,targets = train_images(metadata, embedded)

import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

example_idx =12

def display_image_prediction(model, metadata, embedded, test_idx, example_idx):
    example_image = load_image(metadata[test_idx][example_idx].image_path())
    example_prediction = model.predict([embedded[test_idx][example_idx]])
    example_identity = encoder.inverse_transform(example_prediction)[0]

    # Detect face and return bounding box
    bb = alignment.getLargestFaceBoundingBox(example_image)

    plt.imshow(example_image)
    plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
    plt.title(f'Recognized as {example_identity}')

display_image_prediction(knn, metadata, embedded, test_idx, example_idx)

# Load custom images
import copy

custom_metadata = load_metadata("custom")
metadata2 = np.append(metadata,custom_metadata)
embedded2 = copy.deepcopy(embedded)


for i, m in enumerate(custom_metadata):
    #print("loading image from {}".format(m.image_path()))
    img = load_image(m.image_path())
    vector = get_face_vector(img)
    vector = vector.reshape(1,128)
    embedded2 = np.append(embedded2, vector,axis=0)

encoder,knn,svc,test_idx,targets = train_images(metadata2,embedded2)

import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

example_idx = int(list(targets).index("Brad_Pitt")/2)
display_image_prediction(knn, metadata2, embedded2, test_idx, example_idx)

# Recognize and label unknown images

def display_cv2_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
def label_cv2_image_faces(rgb_img, face_bbs, identities):    
    # Convert RGB back to cv2 RBG format
    img = rgb_img[:,:,::-1]

    for i, bb in enumerate(face_bbs):
        # Draw bounding rectangle around face
        cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(img, (bb.left(), bb.bottom() - 35), (bb.right(), bb.bottom()), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, identities[i], (bb.left() + 6, bb.bottom() - 6), font, 1.0, (255, 255, 255), 1)
    return img

def display_cv2_image_faces(rgb_img, face_bbs, identities):
    img = label_cv2_image_faces(rgb_img, face_bbs, identities)
    display_cv2_image(img)
    
def display_plt_image_faces(img, face_bbs, identities, subplot=111):
    plt.subplot(subplot)
    plt.imshow(img)
    for bb in face_bbs:
        plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
    # TODO: Print identities in correct order
    plt.title(f'Recognized as {identities}')

def identify_image_faces(svc, knn, example_image):
    vectors = get_face_vectors(example_image)
    
    identities = []
    for vector in vectors:
        vector = vector.reshape(1,128)
        confidence_scores = svc.decision_function(vector)    
        if (confidence_scores.max() < 0):
            example_identity = "Unknown"
        else:
            example_prediction = knn.predict(vector)
            example_identity = encoder.inverse_transform(example_prediction)[0]
        identities.append(example_identity)
        
    # Detect faces and return bounding boxes
    face_bbs = alignment.getAllFaceBoundingBoxes(example_image)
    
    return face_bbs, identities
    
def display_unknown_image(svc, knn, image_path, subplot=111):
    example_image = load_image(image_path)
 
    face_bbs, identities = identify_image_faces(svc, knn, example_image)
    #display_cv2_image_faces(example_image, face_bbs, identities)
    display_plt_image_faces(example_image, face_bbs, identities, subplot)

    return example_image

mage1 = display_unknown_image(svc, knn, "unknown/unknown5.jpg", 121)


# Dataset visualization

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(embedded2)

for i, t in enumerate(set(targets)):
    idx = targets == t
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)   

plt.legend(bbox_to_anchor=(1, 1));



