# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:53:33 2018

@author: Fahim
"""

from tkinter import *
from tkinter import filedialog
import glob
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from PIL import ImageTk,Image
import numpy as np

def extract_surf_feature(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoint = surf.detect(image)
    keypoint, descriptors = surf.compute(image,keypoint)
    descriptors = descriptors.flatten()
    descrip = descriptors[0:9856]

    return descrip


labels = []

imageLabel = open('./label.txt', 'r')
#print(imageLabel)
for label in imageLabel:
    label = int(label.strip('\n'))
    labels.append(label)
    
model = KNeighborsClassifier(n_neighbors=3)    

master = Tk()
master.title('Classifier')
master.geometry('800x520')



msg = Message(master, text="")

label = Label(master, text="")
label1 =  Label(master, text="")

var = '/*.jpg'
imagePaths = ''

def OpenFolder():
    global imagePaths
    global msg
    global label
    global label1
    
    try:
        msg.pack_forget()
        label.pack_forget()
        label1.pack_forget()
    except Exception:
        pass
    
    dirname = filedialog.askdirectory(parent=master, initialdir=os.getcwd(), title="Select Folder")
    imagePaths = dirname + var
    
    label = Label(master, text="Folder Opening Done", bg="green", fg="black")
    label.pack()
    msg = Message(master, text=imagePaths)

    msg.pack()


features = []
def Extract_Feature_and_store_in_Database():
    global features
    global imagePaths
    global msg
    global label
    global label1
    
    try:
        label.pack_forget()
        msg.pack_forget()
        label1.pack_forget()
    except Exception:
        pass


    imagefile = glob.glob(imagePaths)
    
    for imagePath in imagefile:
        #print(imagePath)
        image = cv2.imread(imagePath)
        surf = extract_surf_feature(image)
        features.append(surf)
    print(type(features))
    print(features)
    print(type(features))
    df = pd.DataFrame(data=features)
    feature = pd.ExcelWriter('./feature.xlsx', engine='xlsxwriter')
    df.to_excel(feature, sheet_name='Sheet1')
    feature.save()

    label = Label(master, text="Feature extraction completed", bg="green", fg="black")
    label.pack()

featuref = []
def Load_Feature_Dataset():
    global model
    global features
    global labels
    global featuref
    global label
    global msg
    global label1
    
    
    try:
        label.pack_forget()
        msg.pack_forget()
        label1.pack_forget()
    except Exception:
        pass
    
    excelpath = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Feature data",
                                           filetypes=[("Excel File", "*.xlsx")])
    ff = pd.read_excel(excelpath,sheetname = 0, index = False)
    f = ff.values.tolist()
    for i in range(len(f)):
        n = []
        for j in range(len(f[i])):
            n.append(f[i][j])
        featuref.append(n)
        
    r = np.array(featuref)
    g = np.asarray(r, dtype=np.float32)
    featuref = list(g)
    
    
    

    label = Label(master, text="Feature Load completed", bg="green", fg="black")
    label.pack()
    
def Train_KNN():
    global model
    global features
    global labels
    global featuref
    global label
    global msg
    global label1
    
    
    try:
        label.pack_forget()
        msg.pack_forget()
        label1.pack_forget()
    except Exception:
        pass
    
    model.fit(featuref, labels)

    label = Label(master, text="KNN training completed", bg="green", fg="black")
    label.pack()
    
impath = ''

def Input_Query_Image():
    global impath
    global label
    global label
    global msg
    global label1
    
    
    try:
        msg.pack_forget()
        label.pack_forget()
        label1.pack_forget()
    except Exception:
        pass
    
    impath = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Testing Image",
                                           filetypes=[("Image Files", "*.jpg")])
    
    im = cv2.imread(impath)
    cv2.imshow('Query Image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    label = Label(master, text="Query image imported", bg="green", fg="black")
    label.pack()
    msg = Message(master, text=impath)
    msg.pack()

    
    


def Predict_Class():
    global msg
    global label
    global impath
    global model
    global label1
    
    
    try:
        msg.pack_forget()
        label.pack_forget()
        label1.pack_forget()
    except Exception:
        pass
    
    
    queyimage = cv2.imread(impath)
    surf = extract_surf_feature(queyimage)
    result = model.predict([surf])
    val = ''
    if(result[0] == 1):
        val = 'Bus'
    elif(result[0] == 2):
        val = 'Dinosaur'
    elif(result[0] == 3):
        val = 'Flower'
    
    label = Label(master, text="Predicting Class", bg="green", fg="black")
    label.pack()
    label1 = Label(master, text=val, bg="black", fg="yellow")
    label1.config(font=("Courier", 44))
    label1.pack()



b = Button(master, text="Open Training Folder", command=OpenFolder)
b.pack(side="top", padx=4, pady=4)
b = Button(master, text="Extract Feature and store in Database", command=Extract_Feature_and_store_in_Database)
b.pack(side="top", padx=4, pady=4)
b = Button(master, text="Load Feature Dataset", command=Load_Feature_Dataset)
b.pack(side="top", padx=4, pady=4)
b = Button(master, text="Train KNN", command=Train_KNN)
b.pack(side="top", padx=4, pady=4)
b = Button(master, text="Input Query Image", command=Input_Query_Image)
b.pack(side="top", padx=4, pady=4)
b = Button(master, text="Predict Class", command=Predict_Class)
b.pack(side="top", padx=4, pady=4)

mainloop()
