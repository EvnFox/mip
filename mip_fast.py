
from PIL import Image 
import numpy as np 
import time 



if __name__ == "__main__": 

    start_time = time.time()

# Get image and static mask
    pth = "002-001-2023-11-27_17-37-00_IMG-TLP.jpg"
    
    
    sm = np.array(Image.open('static_mask.png'))
    im0 = np.array(Image.open(pth))
    sm = sm.astype('uint8')

#apply static mask to image. 
    im1 = im0[:,:,0]*sm
  
    im = im1 > 75
    im = im.astype('uint8')
# get binary image
    #im = morphology.binary_opening(im)
    #im = morphology.remove_small_objects(im, 5, connectivity=2)

    print(im.sum())
   # im = morphology.remove_small_holes(im)

#img = plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    
#Save imgae.
 #   result = Image.fromarray((im*255).astype(np.uint8))
#    result.save('MIP_out.jpg')

#get properties
    #df, regions_dict = particle_props(im, im0)
    #df.to_csv('MIP_out.csv')

    print(time.time()-start_time)


