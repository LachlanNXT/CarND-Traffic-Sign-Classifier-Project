# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:56:31 2017

@author: Lachlan
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img1=mpimg.imread('./data/Ready/20.jpg')
img2=mpimg.imread('./data/Ready/70.jpg')
img3=mpimg.imread('./data/Ready/keepright.jpg')
img4=mpimg.imread('./data/Ready/noentry.jpg')
img5=mpimg.imread('./data/Ready/padestrian.jpg')
img6=mpimg.imread('./data/Ready/roadwork.jpg')
img7=mpimg.imread('./data/Ready/stop.jpg')
img8=mpimg.imread('./data/Ready/roundabout.jpg')
img9=mpimg.imread('./data/Ready/yield.jpg')

y_images = [0,4,38,17,27,25,14,40,13]
X_images = np.stack((img1,img2,img3,img4,img5,img6,img7,img8,img9))
plt.imshow(img1)

n_images = len(X_images)
n_labels = len(y_images)
images_shape = X_images.shape
print("Number of images =", n_images)
print("Number of labels =", n_labels)
print("Image data shape =", images_shape)
assert(n_images==n_labels)




# TODO: How many unique classes/labels there are in the dataset.
[hist, bin_edges] = np.histogram(y_images,[i for i in range(43)])
plt.hist(y_images,bin_edges)
plt.title("Number of validation labels by type")
for i in bin_edges:
    if hist[i] == 0:
        n_classes = i
        break
        

print("Number of classes =", n_classes)