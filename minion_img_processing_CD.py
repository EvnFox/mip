#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 8 11:05:04 2024

@author: yxs
"""


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob, os, sys

import time

st = time.time()

file = 'minion_pics/002-001-2023-11-27_17-37-00_IMG-TLP.jpg'
image=cv2.imread(file)
# static_mask = cv2.imread('static_mask.png',cv2.IMREAD_GRAYSCALE)
# _, binary_mask = cv2.threshold(static_mask, 128, 255, cv2.THRESH_BINARY)
# image = cv2.bitwise_and(image,image,mask = binary_mask)

img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# cv2.imwrite('081-001_CD_step1.jpg',img_hsv)
# cv2.imwrite('081-001_CD_step2.jpg',img_lab)

# Blur the saturation channel to remove noise
img_hsv[:, :, 1] = cv2.GaussianBlur(img_hsv[:, :, 1], (15, 15), 0)

# cv2.imwrite('081-001_CD_step3.jpg',img_hsv[:, :, 1])

    # Run CLAHE on the image luminance channel
clahe=True
if clahe:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    filtered_lum = clahe.apply(img_lab[:, :, 0])
else:
    filtered_lum = img_lab[:, :, 0]

# cv2.imwrite('081-001_CD_step4.jpg',filtered_lum)

filtered_saturation = img_hsv[:, :, 1]
img_sl = cv2.merge((filtered_saturation, filtered_lum))

# cv2.imwrite('081-001_CD_step5.jpg',img_sl[:,:,1])

# Initialize an empty saliency map
saliency_map = np.zeros_like(image[:, :, 0], dtype=np.float32)

blurred_saturation = cv2.GaussianBlur(img_sl[:, :, 1], (5, 5), 0)
blurred_lum = cv2.GaussianBlur(img_sl[:, :, 1], (5, 5), 0)

# cv2.imwrite('081-001_CD_step6.jpg',blurred_saturation)

# Calculate the center and surround regions for each channel
center_lum = cv2.GaussianBlur(blurred_lum, (3, 3), 2)
surround_lum = blurred_lum - center_lum

# cv2.imwrite('081-001_CD_step7.jpg',surround_lum)

center_saturation = cv2.GaussianBlur(blurred_saturation, (3, 3), 2)
surround_saturation = blurred_saturation - center_saturation

# Normalize the values to the range [0, 255]
center_lum = cv2.normalize(center_lum, None, 0, 255, cv2.NORM_MINMAX)
surround_lum = cv2.normalize(surround_lum, None, 0, 255, cv2.NORM_MINMAX)

center_saturation = cv2.normalize(center_saturation, None, 0, 255, cv2.NORM_MINMAX)
surround_saturation = cv2.normalize(surround_saturation, None, 0, 255, cv2.NORM_MINMAX)


# cv2.imwrite('081-001_CD_step8.jpg',center_saturation)
# cv2.imwrite('081-001_CD_step9.jpg',surround_saturation)


# Combine the center and surround regions for each channel
# Reduce the weight of the saturation channel
saliency_lum = 1.2*(center_lum - surround_lum)
saliency_saturation = (center_saturation - surround_saturation) / 4

# cv2.imwrite('081-001_CD_step10.jpg',saliency_lum)
# cv2.imwrite('081-001_CD_step11.jpg',saliency_saturation)


# Combine the saliency maps into the final saliency map
saliency_level = (saliency_saturation + saliency_lum) / 2

# Resize the saliency map to the original image size
saliency_level = cv2.resize(saliency_level, (image.shape[1], image.shape[0]))

# cv2.imwrite('081-001_CD_step12.jpg',saliency_level)

# Accumulate the saliency map at each level
saliency_map += saliency_level
# Normalize the final saliency map
saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)

# Invert the saliency map so that the salient regions are white
saliency_map_inv = np.abs(saliency_map-255)

# Blur the saliency map to remove noise
saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)

# cv2.imwrite('081-001_CD_step13.jpg',saliency_map)

m= cv2.adaptiveThreshold(saliency_map.astype(np.uint8),
255,  # Max pixel value
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY,
27,  # Block size (size of the local neighborhood)
-2  # Constant subtracted from the mean
)

cv2.imwrite('081-001_CD_step14.jpg',m)

m2=np.zeros_like(m)
contours, hierarchy = cv2.findContours(m, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
cv2.drawContours(m2,contours,-1,50,-1)

cv2.imwrite('081-001_CD_step15.jpg',m2)

edges=cv2.Canny(image.astype(np.uint8),50,60)
kernel = np.ones((5, 5), np.uint8)
edges=cv2.dilate(edges,kernel,2)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
edges2=np.zeros_like(edges)
cv2.drawContours(edges2,contours,-1,255,-1)
edges2[saliency_map<2*np.mean(saliency_map)]=255
kernel = np.ones((15, 15), np.uint8)
edges2=cv2.erode(edges2,kernel,1)
edges2=cv2.erode(edges2,kernel,1)
edges2=cv2.erode(edges2,kernel,1)
edges2=cv2.erode(edges2,kernel,1)
contours, hierarchy = cv2.findContours(edges2, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
edges3=np.zeros_like(edges)
cv2.drawContours(edges3,contours,-1,255,-1)


cv2.imwrite('081-001_CD_step16.jpg',edges)
cv2.imwrite('081-001_CD_step17.jpg',edges2)
cv2.imwrite('081-001_CD_step18.jpg',edges3)

markers=m2.copy()
Canny_edges=cv2.Canny(image.astype(np.uint8),120,130) ##
markers[Canny_edges==255]=255
markers[edges3==0]=0
kernel = np.ones((5, 5), np.uint8)
markers=cv2.dilate(markers,kernel,1)
markers2=np.zeros_like(markers)
# findcontours works on binarized images
contours, hierarchy = cv2.findContours(markers, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
cv2.drawContours(markers2,contours,-1,255,-1)
# cv2.imwrite('081-001_CD_step20_0.jpg',markers2)
markers2=cv2.erode(markers2,kernel,1)
markers2=cv2.erode(markers2,kernel,1)

# cv2.imwrite('081-001_CD_step19_0.jpg',Canny_edges)
# cv2.imwrite('081-001_CD_step19.jpg',markers)
# cv2.imwrite('081-001_CD_step20.jpg',markers2)

contours3, _ = cv2.findContours(markers2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw the contours on the image with a purple line border thickness of 2 px
contour_img = image.copy()
cv2.drawContours(contour_img, contours3, -1, (180, 105, 255), 2)

# cv2.imwrite('081-001_CD_step21.jpg',contour_img)

img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
#using Matplotlib, ensuring scaling is from 0 to 255

print(time.time() - st)

cv2.imwrite('081-001_CD.jpg',contour_img)
