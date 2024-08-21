import numpy as np 
from skimage import transform, measure, morphology
import pandas as pd
import warnings
import os
from PIL import Image 
import cv2 as cv
import matplotlib.pyplot as plt


def add_zeros(x : np.array) -> np.array: 
    ## This function helps mat the scikit-image transform.Intergral_Image look like the matlab IntegralImage.
    z1 = np.zeros([1, np.shape(x)[1]], dtype='uint8')
    z2 = np.zeros([np.shape(x)[0]+1, 1], dtype='uint8')
    x = np.append(z1, x, axis=0)
    x = np.append(z2, x, axis=1)
    return x


def filter_mats(A : np.array, B : np.array) -> np.array: 
    '''
    Takes two matrices of the same shape and and returns a vector
    containing all A[i][j] with A[i][j] < A[i][j]
    '''
    if np.shape(A) != np.shape(B): 
        print("shape mismatch, ensure A and B have the same Shape")
        return 1
    x = list()
    for i in range(np.shape(A)[0]-1):
        for j in range(np.shape(A)[1]-1):
            if A[i][j].dtype == 'StrDType': 
                continue
            if A[i][j] < B[i][j]:
                x.append(A[i][j])

    return np.array(x)

def binerize(A : np.array, B : np.array) -> np.array: 
    """
    Takes two matrices A,B, of the same shape and returns a binary matrix C of the same size 
    where C[i][j] = 1 if A[i][j] > B[i][j] and C[i][j] =0 otherwise. 
    """
    if np.shape(A) != np.shape(B): 
        print("shape mismatch, ensure A and B have the same Shape")
        return 1
    C= np.zeros(np.shape(A))
    for i in range(np.shape(A)[0]-1):
        for j in range(np.shape(A)[1]-1):
            if A[i][j] > B[i][j]: 
                C[i][j] = 1
            else: 
                C[i][j] = 0

    return C


def adaptive_mean(im, sm, *argv, **kwargs): 
    channel = 'r' 
    method = 'edge' 
    CHANNELS = ['r', 'R', 'g', 'G', 'b', 'B']
    METHODS = ['edge', 'constant', 'symmetric']

    #add error handeling. sm would be better than im.
    
    sze = np.floor(np.array(np.shape(im[:,:, 0]))/16)
    nbhd_size = np.array(list(map(int, 2*sze + 1)))
    #print(nbhd_size)
    pad_size = np.array(list(map(int, sze)))
    #print(pad_size)
    for arg in argv: 
        if arg in CHANNELS: 
            channel = arg.lower()
        if arg in METHODS: 
            method = arg 
    


    if channel == 'r': 
        im_r = im[:,:,0] 
    elif channel == 'g': 
        im_r = im[:,:,1] 
    else: 
        im_r = im[:,:,2]
    
 #   match channel:
 #       case 'r': 
  #          im_r = im[:, :, 0] 
  #      case 'g': 
  #          im_r = im[:, :, 1] 
  #      case 'b': 
  #          im_r = im[:, :, 2] 

    im_size = np.shape(im_r)

    # The way matlabs padarray and np.pad take size are different. padarray(x, [3, 2], 'arg') = np.pad(x, [[3,],[2,]], 'arg')
    im_pad = np.pad(im_r, [[pad_size[0],], [pad_size[1],]] , mode =method) 
    mask_pad = np.pad(sm, [[pad_size[0],],[pad_size[1],]], mode =method)
    #print(np.shape(im_pad))
    #print(np.shape(mask_pad))
    # matlab returns an integral image with column 1 and row 1 both being all zeros, but scikit-image does not do this, thus the need for the add_zeros function. 
    int_im = add_zeros(transform.integral_image(im_pad))
    int_mask = add_zeros(transform.integral_image(mask_pad))



    im_sum = int_im[nbhd_size[0]:, nbhd_size[1]:] + int_im[0:im_size[0], 0:im_size[1] ] - int_im[0:im_size[0],nbhd_size[1]:] - int_im[nbhd_size[0]:, 0:im_size[1]]
    mask_sum = int_mask[nbhd_size[0]:, nbhd_size[1]:] + int_mask[0:im_size[0], 0:im_size[1] ] - int_mask[0:im_size[0], nbhd_size[1]:] - int_mask[nbhd_size[0]:, 0:im_size[1]]


    with warnings.catch_warnings(action="ignore"):
       ret = (im_sum/mask_sum ) * sm
    return ret



def bg_lighting_iterations(im, sm, im_mean, *argv, **kwargs): 
    bg_int = np.array([], dtype='double')
    im_red = im[:, :, 0] 

    im_red = np.array([list(map(lambda k: 0.01 if k == 0 else k, x)) for x in im_red])
        
    im_red_masked = im_red*sm 
    #im_red_masked = np.array(list(map(lambda k: np.array(list(map(lambda x: 'nan' if x == 0 else x, k))), im_red_masked)))

        
    #bg_int[i] = np.mean(np.array(list(filter(lambda x: x < 2*im_mean[0], im_red_masked))))

    bg_int = np.mean(filter_mats(im_red_masked, 2.5*im_mean[0]))
      
    return bg_int 



def particle_props(im : np.array, intensity_im : np.array, PROPS : list ) -> pd.DataFrame: 
    label_img = measure.label(im, connectivity=2)
    regions = measure.regionprops_table(label_img, intensity_image=intensity_im, properties=PROPS)
    df = pd.DataFrame(regions) 
    return df, regions

def get_contour(im0 : np.array, im : np.array ) -> np.array: 
    '''
    takes image pth and saves contoured image
    '''

    cpy = im0.copy()
    contours, hierarchy = cv.findContours(im.astype('uint8'), cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)
    cv.drawContours(cpy, contours, -1, (0,255,0), 3)
    return cpy
    


def plot_bbox(im0 : np.array, df : pd.DataFrame): 
    
    plt.figure()
    for i in range(np.shape(df)[0]-1): 
        x = np.array(df.iloc(0)[i])[:-1]
        minr = x[0] 
        minc = x[1]
        maxr = x[2] 
        maxc = x[3]
    #flipping x and y, wanted: [maxr, minc],[ maxr, maxc], [minr, maxc], [minr, minc], there is wired coords, x,y flipped
        coords  = [[maxc, minr],[ maxc, maxr], [minc, maxr], [minc, minr]]
        coords.append(coords[0])
        xs, ys = zip(*coords)
        plt.plot(xs, ys, linewidth=.1)
    plt.axis('off')
    plt.imshow(im0)
    plt.show()
#plt.show()


    







    
