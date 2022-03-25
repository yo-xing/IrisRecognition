import os
import cv2
import numpy as np

def FeatureExtraction(img_enhanced):
    def build_filters():
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters
    
    def process(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum
    filters = build_filters()
    res1 = process(img_enhanced, filters)
    res1_cropped = res1[0:int(3*(res1.shape[0]/4))]
    
    height,width = res1_cropped.shape
    
    num_cols = int(np.ceil(width/8))
    num_rows = int(np.ceil(height/8))
    means = np.zeros((num_rows,num_cols))
    features = []
    sd = np.zeros((num_rows,num_cols))
    row = 0
    y_u = 0

    
    while y_u < height:
        y_d = y_u + 7
        if y_d > height - 1:
            y_d = height - 1

        col = 0
        x_l = 0
        while x_l < width:
            x_r = x_l + 7
            if x_r > width - 1:
                x_r = width - 1

            # loop the row snd columns of the 16x16 blocks to compute the mean value of all pixels in the block
            mean = 0
            sd_sum = 0
            num_pixels = 0
            for c in range(x_l,x_r + 1):
                for r in range(y_u,y_d + 1):
                    num_pixels += 1
                    mean += res1_cropped[r,c]

            # store block's mean value is matrix
            means[row,col] = mean/num_pixels
            features.append(means[row,col])
            for c in range(x_l,x_r + 1):
                for r in range(y_u,y_d + 1):
                    sd_sum += (res1_cropped[r,c]-means[row,col])**2

            # store block's sd value is matrix        
            sd[row,col] = (sd_sum/num_pixels)**0.5
            features.append(sd[row,col])
            x_l = x_r + 1
            col += 1
        y_u = y_d + 1
        row += 1
    return(features)
