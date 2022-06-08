"""
Copyright © Nima Sedaghat 2017-2021

All rights reserved under the GPL license enclosed with the software. Over and
above the legal restrictions imposed by this license, if you use this software
for an academic publication then you are obliged to provide proper attribution
to the below paper:

    Sedaghat, Nima, and Ashish Mahabal. "Effective image differencing with
    convolutional neural networks for real-time transient hunting." Monthly
    Notices of the Royal Astronomical Society 476, no. 4 (2018): 5365-5376.
"""

import numpy as np
from skimage import io,img_as_float
from math import cos,sin,pi
import cv2

def warp_crop_direct(src_img,base_crop_width,base_crop_height,scale,rotation,translation,from_center=True, symmetric=False):

    x1 = 0-(base_crop_width//2)
    y1 = 0-(base_crop_height//2)
    x2 = x1+base_crop_width-1
    y2 = y1+base_crop_height-1

    src_points = np.array([
        [x1, y1],
        [x1, y2],
        [x2, y2],
        [x2, y1]
    ], dtype="float32")

    if from_center:
        xc = src_img.shape[1]//2
        yc = src_img.shape[0]//2
        src_points += [xc,yc]

    
    theta = -1 * rotation*pi/180
    T = np.array([
        [cos(theta) , -sin(theta)],
        [sin(theta) , cos(theta)]
    ], dtype="float32")/scale

    #--- Apply the transform
    src_points = (T @ src_points.T).T - np.floor(translation)  #Rounding to be compatible with the old transformation method
    src_points = np.float32(np.round(src_points)) #Rounding to be compatible with the old transformation method

    # the perspective transformation matrix
    dst_points = np.array([ #create points up-side-down to account for the y-mirroring in opencv
        [0, 0],
        [0, base_crop_height],
        [base_crop_width, base_crop_height],
        [base_crop_width, 0],
    ], dtype="float32")

    
    M = cv2.getPerspectiveTransform(src_points, dst_points)


    if not symmetric:
        return cv2.warpPerspective(src_img, M, (base_crop_width, base_crop_height),flags=cv2.INTER_NEAREST)
    else:
        return cv2.warpPerspective(src_img, M, (base_crop_width, base_crop_height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)


if __name__ == '__main__':
    filename = '/home/nima/Pictures/snr.png'
    img = img_as_float(io.imread(filename,as_gray=True))

    crop = warp_crop_direct(img,256,256,scale=2,rotation=45,translation=[0,100])

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(crop)
    plt.show()
