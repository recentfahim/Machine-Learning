# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 02:36:23 2018

@author: Fahim
"""

import glob
import cv2
from sklearn import svm
from sklearn.externals import joblib

des = []
def extract_surf_feature(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoint = surf.detect(image)
    keypoint, descriptors = surf.compute(image, keypoint)
    descriptors = descriptors.flatten()
    #print(len(descriptors))
    des.append(len(descriptors))
    descrip = descriptors[0:27100]

    return descrip

label = []
feature = []
images = glob.glob("Training/One/*.jpg")
count = 1
for i in images:
    print(i)
    img = cv2.imread(i)
    surf = extract_surf_feature(img)
    feature.append(surf)
    label.append(count)
#print(label)
print(len(label))
#print(feature)

images = glob.glob("Training/Two/*.jpg")
count = 2
for i in images:
    print(i)
    img = cv2.imread(i)
    surf = extract_surf_feature(img)
    feature.append(surf)
    label.append(count)
#print(label)
print(len(label))
#print(feature)

images = glob.glob("Training/Five/*.jpg")
count = 3
for i in images:
    print(i)
    img = cv2.imread(i)
    surf = extract_surf_feature(img)
    feature.append(surf)
    label.append(count)
#print(label)
print(len(label))
#print(feature)

images = glob.glob("Training/Ten/*.jpg")
count = 4
for i in images:
    print(i)
    img = cv2.imread(i)
    surf = extract_surf_feature(img)
    feature.append(surf)
    label.append(count)
#print(label)
print(len(label))
#print(feature)

model = svm.SVC()
model.fit(feature, label)


print("Train Finished")

joblib.dump(model, 'Model/weightssvm.pkl')