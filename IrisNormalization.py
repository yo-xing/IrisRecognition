# import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def IrisNormalization(img_gray,xy,radii):
    
# parameters:
# img_iris: the segmented iris image resulting from IrisLocalization
# radii: an array storing the radius of each boundary circle (pupil and iris, respectively)
# centers: an array storing the centers of each boundary circle (pupil and iris, respectively)

# return:
# img_normalized: the 64x512 normalized iris image "unrolled" counterclockwise using polar coordinates

# function: convert the gray iris image to polar coordinates for iris normalization

    # initialize a two-dimensional matrix for the normalized image using fixed dimensions 64x512 (as noted in the Li Ma paper)
    height = 64
    circum = 512
    img_normalized = np.zeros((height,circum), np.uint8)
    
    # convert to polar coordinates
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / circum)
    for i in range(circum):
        theta = thetas[i]  # value of theta coordinate
        # get coordinate of inner (i) and outer (o) boundaries
        # note: only the center of the pupil boundary circle is used; the distance between the two circles' centers is negligible
        Xi = xy[0][0] + radii[0] * np.cos(theta)
        Yi = xy[0][1] + radii[0] * np.sin(theta)
        Xo = xy[0][0] + radii[1] * np.cos(theta)
        Yo = xy[0][1] + radii[1] * np.sin(theta)
    
        for j in range(height):
            # if the iris image is cropped, fill in with zero pixel values (no features present to be detected)
            if (Yo < img_gray.shape[0] and Yo >= 0) and (Xo < img_gray.shape[1] and Xo >= 0):            
                Xc = Xi + (Xo-Xi) * (j/height)  # j/irisHeight is value of r coordinate (normalized)
                Yc = Yi + (Yo-Yi) * (j/height)
                img_normalized[j][i] = img_gray[int(Yc)][int(Xc)]  # intensity of the pixel
            else:
                img_normalized[j][i] = 0
    
    # return the normalized image
    return img_normalized