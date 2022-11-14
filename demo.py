
#%%

import cv2
import numpy as np
from matplotlib import pyplot as plt
from misc_func import get_corners,extra_points,mse,rotate_and_correct

img = cv2.imread('distorted.png')
im2 = cv2.imread('real.png')


corners=get_corners(img)

icorners = []
for corner in corners:
    pt = [ corner[0][0],corner[0][1] ]
    icorners.append(pt)
icorners=sorted(icorners,key=lambda x:(x[0],x[1]))
icorners=sorted(icorners[:2],key=lambda x:x[1])+sorted(icorners[-2:],key=lambda x:x[1])
icorners = np.float32(icorners)
'''icorners=extra_points(icorners)'''


length=0.25*(np.linalg.norm(icorners[0]-icorners[1])+np.linalg.norm(icorners[2]-icorners[1])+np.linalg.norm(icorners[2]-icorners[3])+np.linalg.norm(icorners[0]-icorners[3]))
length=int(np.float32(length))

ocorners = [ [0,0], [0,length],[length,0],[length,length] ]
ocorners = np.float32(ocorners)
'''ocorners=extra_points(ocorners)'''

h, mask = cv2.findHomography(icorners, ocorners, cv2.RANSAC)
M = cv2.getPerspectiveTransform(icorners, ocorners)

image1 = cv2.warpPerspective(img, h, (length, length)) 
image2 = cv2.warpPerspective(img, M, (length, length)) 

im1Reg=rotate_and_correct(im2,image1)
im2Reg=rotate_and_correct(im2,image2)
print(mse(im2,im1Reg),mse(im2,im2Reg))
cv2.imwrite("Registered.png", im1Reg)
cv2.imwrite("Registered2.png", im2Reg)

'''print(mse(im2,image1),mse(im2,image2))
cv2.imwrite("Registered.png", image1)
cv2.imwrite("Registered2.png", image2)'''





