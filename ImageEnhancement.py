# import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ImageEnhancement(img_irisNormalized):
    
# parameters:
# img_irisNormalized: the 64x512 normalized iris image produced by IrisNormalization

# return:
# img_irisEnhanced: the normalized iris image after image enhancement

# function: detect and remove glare and eyelashes from the normalized iris using thresholding absed off the histogram, then enhance the image using background illumination removal and histogram equalization over blocks of the normalized iris image

    # part 1: glare and eyelash detection and removal
    # comment out to include
    # it is excluded here because it does not improve the quality of the enhancement, but rather results in block boundaries that act as false features
    
#     # create masks for glare and eyelashes and combine using bitwise_or
#     mask_glare = cv2.inRange(img_irisNormalized, 200, 255)
#     mask_lashes = cv2.inRange(img_irisNormalized, 0, 100)
#     mask_noise = cv2.bitwise_or(mask_glare, mask_lashes)
    
#     # invert the mask and apply over normalized iris image using bitwise_and
#     mask_noise = np.invert(mask_noise)
#     img_irisNormalized = cv2.bitwise_and(img_irisNormalized, img_irisNormalized, mask = mask_noise)

    # part 2: image enhancement
    # obtain the dimensions of the normalized image
    height,circum = img_irisNormalized.shape

    # determine the dimensions of and initialize the matrix to store the mean pixel values of adjacent 16x16 pixel blocks traversing the normalized image
    num_rows = int(np.ceil(height/16))
    num_cols = int(np.ceil(circum/16))
    means = np.zeros((num_rows,num_cols))

    # loop to traverse the normalized image horizontally and vertically , obtaining and storing the means of the values in each section
    # note: the code allows for blocks that are smaller than 16x16, though this does not occur with our dimensions
    row = 0
    y_u = 0
    while y_u < height:
        y_d = y_u + 15
        if y_d > height - 1:
            y_d = height - 1

        col = 0
        x_l = 0
        while x_l < circum:
            x_r = x_l + 15
            if x_r > circum - 1:
                x_r = circum - 1

            # loop the rows and columns of the 16x16 blocks to compute the mean value of all pixels in the block
            mean = 0
            num_pixels = 0
            for r in range(y_u,y_d + 1):
                for c in range(x_l,x_r + 1):
                    mean += img_irisNormalized[r,c]
                    num_pixels += 1

            # store block's mean value in matrix
            means[row,col] = mean/num_pixels

            # shift 16x16 block to the right and move to next place in matrix column
            x_l = x_r + 1
            col += 1
            
        # shift 16x16 block down to start the next row and move to next matrix row
        y_u = y_d + 1
        row += 1
    
    # use bicubic interpolation to convert the resulting mean histogram to the same dimensions as the normalized image (64x512)
    img_means = cv2.resize(means,(circum,height),interpolation = cv2.INTER_CUBIC)
    
    # remove background illumination by subtracting mean matrix from normalized image; convert to uint8 to allow histogram equalization
    img_irisEnhanced = np.uint8(np.around(img_irisNormalized - img_means))
    
    # invert image to correct inversion during mean subtraction and enhancement process
    img_irisEnhanced = np.invert(img_irisEnhanced)
    
    # uncomment for image with histogram equalized over all entire pixel values
    # img_equalizeEntireImage = cv2.equalizeHist(img_irisEnhanced)
    
    # loop to traverse the normalized image horizontally and vertically , equalizing the histograms of each 32x32 block
    # note: the code allows for blocks that are smaller than 32x32, though this does not occur with our dimensions
    row = 0
    y_u = 0
    while y_u < height:
        y_d = y_u + 31
        if y_d > height - 1:
            y_d = height - 1

        col = 0
        x_l = 0
        while x_l < circum:
            x_r = x_l + 31
            if x_r > circum - 1:
                x_r = circum - 1

            # equalize the histogram of the 32x32 block
            img_irisEnhanced[y_u:y_d + 1,x_l:x_r + 1] = cv2.equalizeHist(img_irisEnhanced[y_u:y_d + 1,x_l:x_r + 1])

            # shift 32x32 block down to start the next row
            x_l = x_r + 1
            
        # shift 32x32 block to the right
        y_u = y_d + 1
    
    # return the enhanced image
    return img_irisEnhanced

