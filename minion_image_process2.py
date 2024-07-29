

from PIL import Image 
from skimage import transform, measure, morphology
import pandas as pd
import warnings
import numpy as np 
import time 

#threshold for creating binary image.
THRESH = 2.5
#props to be retrieved by region props. 
PROPS = ['area'] 

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
    containing all A[i][j] with A[i][j] < B[i][j]
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


def adaptive_mean(im, sm, *argv, **kwargs): 
    channel = 'r' 
    method = 'edge' 

    #add error handeling. sm would be better than im.
    
    sze = np.floor(np.array(np.shape(im[:,:, 0]))/16)
    nbhd_size = np.array(list(map(int, 2*sze + 1)))
    #print(nbhd_size)
    pad_size = np.array(list(map(int, sze)))
    #print(pad_size)
  

    im_r = im[:, :, 0] 


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



    ret = (im_sum/mask_sum ) * sm
    return ret

#def remove_edge(im : np.array, sm : np.array) -> np.array:
   # ind = np.array([])

   # labels, n = measure.label(im, connectivity=2, return_num=True)

  #  p_area = np.zeros(n)
  #  for i in range(n): 
  #      p_area[i] = sum([x if x == i else 0 for x in labels ])



def particle_props(im : np.array, intensity_im : np.array ) -> pd.DataFrame: 
    label_img = measure.label(im, connectivity=2)
    regions = measure.regionprops_table(label_img, intensity_image=intensity_im, properties=PROPS)
    df = pd.DataFrame(regions) 
    return df, regions


if __name__ == "__main__": 

    start_time = time.time()

# Get image and static mask
    pth = "minion_pics/002-001-2023-11-27_17-37-00_IMG-TLP.jpg"
    sm = np.array(Image.open('static_mask.png'))
    im0 = np.array(Image.open(pth))
    sm = sm.astype('uint8')

#apply static mask to image. 
    im0 = np.array(list(map(lambda x : x*sm.T, im0.T))).T


#get adapative mean
    im_mean_r = adaptive_mean(im0, sm, 'r', 'edge')
  

# get binary image
    im = THRESH*im_mean_r < im0[:,:,0]
    im = morphology.binary_opening(im)
    im = morphology.remove_small_objects(im, 5, connectivity=2)

    print(im.sum)
   # im = morphology.remove_small_holes(im)

#img = plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    
#Save imgae.
 #   result = Image.fromarray((im*255).astype(np.uint8))
#    result.save('MIP_out.jpg')

#get properties
    #df, regions_dict = particle_props(im, im0)
    #df.to_csv('MIP_out.csv')

    print(time.time()-start_time)


