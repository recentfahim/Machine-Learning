# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 02:40:30 2018

@author: Fahim
"""

import glob
import cv2
from sklearn.externals import joblib


model = joblib.load('Model/weightssvm.pkl')

def extract_surf_feature(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoint = surf.detect(image)
    keypoint, descriptors = surf.compute(image, keypoint)
    descriptors = descriptors.flatten()
    #print(len(descriptors))
    #des.append(len(descriptors))
    descrip = descriptors[0:27100]

    return descrip

def predicts(surf):
    text = ""
    result = model.predict([surf])
    if result[0] == 1:
        text = "One Taka"
    elif result[0] == 2:
        text = "Two Taka"
    elif result[0] == 3:
        text = "Five Taka"
    elif result[0] == 4:
        text = "Ten Taka"
    return text 


testimages = glob.glob("Testing/*.jpg")
print("Predict Class")
for image in testimages:
    img = cv2.imread(image)  # reading testing image
    surf = extract_surf_feature(img)
    # surf = np.asarray(img,dtype = "float32")
    clas = predicts(surf)
    print(clas)
