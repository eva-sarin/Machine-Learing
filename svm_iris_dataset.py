# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:08:38 2020

@author: Dell
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets #read in notes on svm on iris datset

#important parameters fro SVC:gamma and C
#gamma--> defines how far the influence of a single training example reaches
#low value:influencereaches far  High value:influence reaches close

#C --> trades off hyperplane surfacee simplicity + training eg misclassifications
# low value =simple/smooth hyperplane surface
#high value=all trainng eg classified correctly but complex surface

dataset=datasets.load_iris() #to load iris datset saved in sklearn

#print(dataset)

features=dataset.data #to differ feature from target
targetVariables=dataset.target #segregating train and test dataset


featureTrain,featureTest,targetTrain,targetTest= train_test_split(features,targetVariables,test_size=0.3)

model=svm.SVC()
fittedModel=model.fit(featureTrain,targetTrain)
predictions=fittedModel.predict(featureTest)

print(confusion_matrix(targetTest,predictions))
print(accuracy_score(targetTest,predictions))