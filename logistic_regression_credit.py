# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 01:44:46 2020

@author: Dell
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

creditData=pd.read_csv("credit_data.csv")

print(creditData.head()) #shows the top 5 entries of datset
print(creditData.describe()) #shows the most important details of the datset
print(creditData.corr()) #show the correlation between different featues ina matrix form'

features=creditData[["duration_in_month","credit_amount"]]  #hence its a multivalus lgistic regression
target=creditData.default  #this tells the final number of probablity if the payer would payback or not which here  is 0.7

feature_train,feature_test,target_train,target_test=train_test_split(features,target,test_size=0.3)

model=LogisticRegression()
model.fit=model.fit(feature_train,target_train)
predictions=model.fit.predict(feature_test)

print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))