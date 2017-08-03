#/usr/bin/python
#
# Function creates an initial mask that removes horizontal and vertical lines from the raw recording
#
# Development log:
#	- 01-08-2016: Bart Root, initial development
#
#------------------------------------------------------------------------------------------
 
# import other software 
import numpy as np
import PIL.Image
from scipy.misc import *
import cv2

# Start function

def create_mask(I):

   # create row mean and std
   stdy=I.std(axis=0);
   stdx=I.std(axis=1);

   mx = np.array((stdx>stdx.mean()+stdx.max()/20.0).nonzero())
   my = np.array((stdy>stdy.mean()+stdy.max()/20.0).nonzero())

   MX = np.zeros(I.shape)
   MY = np.zeros(I.shape)
   MX[mx,:] = 1
   MY[:,my] = 1

   maskI2 = MX+MY

   (i,j) = (maskI2>1).nonzero()
   maskI2[i,j] = 1

   maskI2 = 1-maskI2

   # Return the combined mask
   mask = maskI2
  
   return mask

def create_mask_v1(I, time_step):
   # Combined filter for horizontal and vertical stripes
   scale = 1./time_step
   indices = I > np.mean(I) + 2*np.std(I)
   maskI1 = np.zeros(I.shape)
   maskI1[indices] = 1
   ## Check hoek
   maskI1 = cv2.erode(maskI1,np.ones((int(5*scale+1),1)))
   maskI1 = 1 - cv2.dilate(maskI1, np.ones((120*scale,50/scale)))
   # Mask for horizontal bars
   maskI2 = create_mask(I)
   # Combined mask
   mask = np.multiply(maskI1, maskI2)
   return mask



def create_mask2(I):
   mask = create_mask(I)
   toimage(mask*I).show()

