from IrisMatching import FeatureProcessing
from IrisMatching import reduce_dim
from IrisMatching import IrisMatching
# import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import fucntions for iris recognition
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction

##

import glob
import math
from scipy.spatial import distance
from sklearn import metrics

def PerformanceEvaluation(correct_L1,correct_L2,correct_cosine):
    #evaluates the CRR for each similarity metric given an output of IrisMatching()
    numRight_L1 = sum(correct_L1)
    numRight_L2 = sum(correct_L2)
    numRight_cosine = sum(correct_cosine)
    
    crr_L1=numRight_L1/len(correct_L1)
    crr_L2= numRight_L2/len(correct_L2)
    crr_cosine= numRight_cosine/len(correct_cosine)
    
    
    return crr_L1*100,crr_L2*100,crr_cosine*100


def GraphPerformance(X_train, y_train, X_test, y_test):
    #obtaining and graphing/recording the CRR for every similarity measure and on each # of components below (PCA not included)
    comps = [10, 40 , 60, 80, 90, 106, 'No Dimensionality Reduction'] #number of components being tested
    scores_dict = dict()
    for i in ['LDA']: #generating score for all number of components
        for j in comps:
            if j == 'No Dimensionality Reduction':
                correct_L1,correct_L2,correct_cosine,correct_cosine_ROC = IrisMatching(
            X_train, y_train, X_test, y_test, 106, 1, method = i)
            else:
                correct_L1,correct_L2,correct_cosine,correct_cosine_ROC = IrisMatching(
            X_train, y_train, X_test, y_test, j, 0, method = i)
            scores_dict[(i, j)] = PerformanceEvaluation(correct_L1,correct_L2,correct_cosine)
            cols = ["correct_recognition_rate_L1", "correct_recognition_rate_L2",
        "correct_recognition_rate_cosine"]
    scores = pd.DataFrame(scores_dict).T #creating data frame of the scores
    scores.columns = cols
    scores.to_csv('EvaluationScores.csv')
    print(scores)
    
    #graphing the CRR against number of compoenents for each similarity measure
    y = scores.values[:6]
    x = np.array([10, 40 , 60, 80, 90, 106])
    y1 = y[:, 0]
    y2 = y[:, 1]
    y3 = y[:, 2]
    fig, ax = plt.subplots() 

    ax.plot(x, y1, label = 'L1')
    ax.plot(x, y2, label = 'L2')
    ax.plot(x, y3, label = 'Cosine')
    ax.legend(loc = 'upper left')
    plt.xlabel("Dimensionality of the Feature Vector")
    plt.ylabel("Correct Recognition Rate")
    plt.show()
    plt.savefig('fig10.png')
    return scores 

def GraphROC(X_train, y_train, X_test, y_test):
    #Graphing the ROC curve
    
    #creating the ROC results with 60 components 
    correct_L1,correct_L2,correct_cosine,correct_cosine_ROC = IrisMatching(X_train, y_train, X_test, y_test, 60, 0, 'LDA')
    
    fmr_all=[]
    fnmr_all=[]

    for q in range(3):
        #calculating the False Non-Match Rates and False Match Rates
        fa=0
        fr=0
        num_1=len([i for i in correct_cosine_ROC[q] if i==1])
        num_0=len([i for i in correct_cosine_ROC[q] if i==0])

        for p in range(len(correct_cosine)):
            if correct_cosine[p]==0 and correct_cosine_ROC[q][p]==1:
                fa+=1
            if correct_cosine[p]==1 and correct_cosine_ROC[q][p]==0:
                fr+=1
        fmr=fa/num_1
        fnmr=fr/num_0
        thresh=[0.4,0.5,0.6]
        fmr_all.append(fmr)
        fnmr_all.append(fnmr)

    #creating a table of the curves: roc_table
    dict1={'Threshold':thresh,'FMR':fmr_all,'FNMR':fnmr_all}
    roc_table=pd.DataFrame(dict1)
    print(roc_table)
    roc_table.to_csv('ROC.csv')

    #Plotting the curves
    plt.plot(fnmr_all,fmr_all)
    plt.title('ROC Curve')
    plt.ylabel('False Non-Match Rate')
    plt.xlabel('False Match Rate')
    plt.show()
    plt.savefig('ROC_Curve.png')