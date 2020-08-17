# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 03:15:31 2020

@author: Dell
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

data=pd.read_csv("credit_data.csv")

#log. Regg. accuracy is 93%
#we do better with knn at 98.5%
#which otheriwse would be some 84%

data.features=data[["duration_in_month","credit_amount"]]  #hence its a multivalus lgistic regression
data.target=data.default  #this tells the final number of probablity if the payer would payback or not which here  is 0.7

data.features=preprocessing.MinMaxScaler().fit_transform(data.features) #huge difference

feature_train,feature_test,target_train,target_test=train_test_split(features,target,test_size=0.3)

model=LogisticRegression()
model.fit=model.fit(feature_train,target_train)
predictions=model.fit.predict(feature_test)

print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))