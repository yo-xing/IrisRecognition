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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



def FeatureProcessing(numEyes = 107, path = "datasets"):
    ###Builds and Processes the Features into Train and Test sets in numpy array form for numEyes amount of eyes###
    
    dataFolder = "CASIA Iris Image Database (version 1.0)"

    features_dict = dict()

    for eyeNum in range(1, numEyes+1):
        if eyeNum <= 9:
            eyeId = "00" + str(eyeNum)
        elif 10 <= eyeNum <= 99:
            eyeId = "0" + str(eyeNum)
        else:
            eyeId = str(eyeNum)
        
        imgPath = os.path.join(path, dataFolder, eyeId, "1")
        for sample in range(1,4):   
            imgFilename = eyeId + "_1_" + str(sample) + ".bmp"
        
            # read image and convert to gray scale
            imgName = os.path.join(imgPath, imgFilename)
            img_org = cv2.imread(imgName)
            img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        
            # pass the gray eye image through the different functions in the iris recognition process
            img_irisLocalized, xy, radii = IrisLocalization(img_gray)
            img_irisNormalized = IrisNormalization(img_gray, xy, radii)
            img_irisEnhanced = ImageEnhancement(img_irisNormalized)
            img_features = FeatureExtraction(img_irisEnhanced)
            features_dict[imgFilename.replace('.bmp', '')] = img_features
     
    df = pd.DataFrame(features_dict).T
    #df.to_csv('img_featues_train.csv')
    
    features_dict = dict()

    for eyeNum in range(1, numEyes+1):
        if eyeNum <= 9:
            eyeId = "00" + str(eyeNum)
        elif 10 <= eyeNum <= 99:
            eyeId = "0" + str(eyeNum)
        else:
            eyeId = str(eyeNum)
        
        imgPath = os.path.join(path, dataFolder, eyeId, "2")
        for sample in range(1,5):   
            imgFilename = eyeId + "_2_" + str(sample) + ".bmp"
        
            # read image and convert to gray scale
            imgName = os.path.join(imgPath, imgFilename)
            img_org = cv2.imread(imgName)
            img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        
            # pass the gray eye image through the different functions in the iris recognition process
            img_irisLocalized, xy, radii = IrisLocalization(img_gray)
            img_irisNormalized = IrisNormalization(img_gray, xy, radii)
            img_irisEnhanced = ImageEnhancement(img_irisNormalized)
            img_features = FeatureExtraction(img_irisEnhanced)
            features_dict[imgFilename.replace('.bmp', '')] = img_features

    df_test = pd.DataFrame(features_dict).T
    cols = [str(x) for x in df_test.columns]
    df_test.columns = cols
    cols = [str(x) for x in df.columns]
    df.columns = cols
    eyeNum = [int(x[:3]) for x in df.index]
    df['eyeNum'] = eyeNum
    eyeNum = [int(x[:3]) for x in df_test.index]
    df_test['eyeNum'] = eyeNum
    df = pd.concat([df, df_test])
    df.to_csv('img_featues.csv')
    x_coordinate = df.values[ :, :-1]
    y_coordinate =  df.values[ :, -1:]

    X_test = x_coordinate[numEyes*3:]
    y_test = y_coordinate[numEyes*3:].reshape(numEyes*4, )

    X_train = x_coordinate[:numEyes*3]
    y_train = y_coordinate[:numEyes*3].reshape(numEyes*3,)
    
    return X_train, y_train, X_test, y_test


def reduce_dim(X_train, y_train, X_test, y_test, num, method = "LDA"):
    ### Reducing the dimensions of the feature vectors into num componenents, using either "LDA" or "PCA" as the methods
    ### "LDA"(Fisher linear discriminant) is the default, however "PCA" (Principal Component Analysis) performs better with small number of components
    
    #fit the LDA or PCA model on training data with n num
    if method == "LDA":
        dimred = LinearDiscriminantAnalysis(n_components=num)
        dimred.fit(X_train,y_train)
    elif method == "PCA":
        dimred = PCA(n_components=num)
        dimred.fit(X_train)
    r_train=dimred.transform(X_train)
    
    r_test=dimred.transform(X_test)
    
    y_pred = []
    if method == "LDA":
        y_pred=dimred.predict(X_test)
    
    return r_train,r_test, y_pred





def IrisMatching(X_train, y_train, X_test, y_test, num, keep_dim, method = "LDA"):
    
    #if keep_dim is 1, dimesionality is not reduced. If it is 0, we use the reduce_dim function
    if keep_dim==1:
        r_train=X_train
        r_test=X_test
        
    elif keep_dim==0:
        r_train,r_test,y_pred=reduce_dim(X_train, y_train, X_test, y_test, num, method)

    
    list_L1=[]
    list_L2=[]
    list_cosine=[]
    min_cosine=[]
    
    for i in range(0,len(r_test)):
        L1=[]
        L2=[]
        Cosine=[]
        
        for j in range(0,len(r_train)):
            f=r_test[i]
            fi=r_train[j]
            sumL1=0 #L1 distance
            sumL2=0 #L2 distance
            cosinedist=0 #cosine distance

            sumL1 = distance.cityblock(f, fi)
            sumL2 = distance.euclidean(f, fi)
            cosinedist = distance.cosine(f, fi)

            L1.append(sumL1)
            L2.append(sumL2)
            Cosine.append(cosinedist)
        list_L1.append(L1.index(min(L1)))
        list_L2.append(L2.index(min(L2)))
        list_cosine.append(Cosine.index(min(Cosine)))
        min_cosine.append(min(Cosine))
        
    match=0
    count=0
    
    correct_L1=[]
    correct_L2=[]
    correct_cosine=[]
    correct_cosine_ROC=[]
    
    thresh=[0.4,0.5,0.6]
    
    for x in range(0,len(thresh)):
        correct_ROC=[]
        for y in range(0,len(min_cosine)):
            if min_cosine[y]<=thresh[x]:
                correct_ROC.append(1)
            else:
                correct_ROC.append(0)
        correct_cosine_ROC.append(correct_ROC)
        
    
    for k in range(0,len(list_L1)):
        if count<4:
            count+=1
        else:
            match+=3
            count=1
            
        if list_L1[k] in range(match,match+3):
                correct_L1.append(1)
        else:
            correct_L1.append(0)
        if list_L2[k] in range(match,match+3):
            correct_L2.append(1)
        else:
            correct_L2.append(0)
        if list_cosine[k] in range(match,match+3):
            correct_cosine.append(1)
        else:
            correct_cosine.append(0)

    return correct_L1,correct_L2,correct_cosine,correct_cosine_ROC

