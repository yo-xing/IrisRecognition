import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction

import glob
import math
from scipy.spatial import distance
from sklearn import metrics

from IrisMatching import FeatureProcessing
from IrisMatching import reduce_dim
from IrisMatching import IrisMatching
from PerformanceEvaluation import PerformanceEvaluation
from PerformanceEvaluation import GraphPerformance
from PerformanceEvaluation import GraphROC


#code for reading the csv directly in instead of generating features (to save time)
#df = pd.read_csv('img_featues.csv', index_col = 'Unnamed: 0')
# x_coordinate = df.values[ :, :-1]
# y_coordinate =  df.values[ :, -1:]

# X_test = x_coordinate[107*3:]
# y_test = y_coordinate[107*3:].reshape(107*4, )

# X_train = x_coordinate[:107*3]
# y_train = y_coordinate[:107*3].reshape(107*3,)


X_train, y_train, X_test, y_test = FeatureProcessing()


#graphing the CRR performance and generating a table of scores 
GraphPerformance(X_train, y_train, X_test, y_test)

print('\n\n\n')
print("We chose 60 as the number of components, as that gave us the highest possible L1 score without dropping the others below 75%")
print('\n\n\n')


#Graphing the ROC curve and generating a table for the False Non-Match Rates and False Match Rates
GraphROC(X_train, y_train, X_test, y_test)
