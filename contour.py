from PIL import Image 
import numpy as np 
#import matplotlib.pyplot as plt 
#from skimage import morphology
import functions as fns
import time 
import os

import cv2 as cv


#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'tk')

THRESH = 2.5

PROPS =['bbox', 'area_bbox', 'area']
st = time.time()

# Get image and static mask
#092-011-2023-11-28_09-02-49_IMG-FIN
#pth = "minion_pics/FIN/092-011-2023-11-28_09-02-49_IMG-FIN.jpg"
pth="minion_pics/002-001-2023-11-27_17-37-00_IMG-TLP.jpg"
#pth = "minion_pics/010-001-2023-11-27_18-57-31_IMG-TLP.jpg"
#pth=  'minion_pics/002-001-2023-11-27_17-37-00_IMG-TLP.jpg'
sm = np.array(Image.open('static_mask.png'))
im0 = np.array(Image.open(pth))
sm = sm.astype('uint8')

#apply static mask to image. 
im0 = np.array(list(map(lambda x : x*sm.T, im0.T))).T

im_mean_r = fns.adaptive_mean(im0, sm, 'r', 'edge')


im = THRESH*im_mean_r < im0[:,:,0]
#im = morphology.binary_opening(im)
#im = morphology.remove_small_objects(im, 10, connectivity=2) 

df, regions_dict = fns.particle_props(im, im0, PROPS=PROPS)


prc = im0.copy()
contours, hierarchy = cv.findContours(im.astype('uint8'), cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)
cv.drawContours(prc, contours, -1, (250, 0, 0), 2)
print(time.time() -st) 

cv.imwrite('test.png', prc) 
print(time.time()-st)


