# import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ImageEnhancement(img_irisNormalized):
    #print("right one")
    img_irisEnhanced = cv2.equalizeHist(img_irisNormalized)
    return img_irisEnhanced

