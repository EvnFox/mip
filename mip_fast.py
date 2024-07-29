
from PIL import Image 
import numpy as np 
import time 

start_time = time.time()

# Get image and static mask
pth = "002-001-2023-11-27_17-37-00_IMG-TLP.jpg"
    
    
sm = np.array(Image.open('static_mask.png'))
sm = sm.astype('uint8')
im0 = np.array(Image.open(pth))


#apply static mask to image. 
im1 = im0[:,:,0]*sm
  
im = im1 > 50
im = im.astype('uint8')


print(im.sum())

print(time.time()-start_time)


