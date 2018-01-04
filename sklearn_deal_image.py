# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:16:30 2018

@author: rd0348
"""

import pandas as pd
import matplotlib.pyplot as plt,matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labels_images=pd.read_csv('digits/train.csv')
images=labels_images.iloc[0:5000,1:]
labels=labels_images.iloc[0:5000,:1]
train_images,test_images,train_labels,test_labels=train_test_split(images,labels,train_size=0.8,random_state=0)

i=1
img=train_images.iloc[i].as_matrix()
imgs=img.reshape((28,28))
plt.imshow(imgs,cmap='gray')
plt.title(train_labels.iloc[i,0])





























