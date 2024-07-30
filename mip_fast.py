from PIL import Image 
import numpy as np 
import time 

start_time = time.time()

pth = "minion_pics/002-001-2023-11-27_17-37-00_IMG-TLP.jpg" 
    
sm = np.array(Image.open('static_mask.png')).astype('uint8')
im0 = np.array(Image.open(pth)).astype('uint8')
 
im = im0[:,:,0]*sm > 60

print(im.sum())

print(time.time()-start_time)


