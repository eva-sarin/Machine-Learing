# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:00:35 2020

@author: Dell
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

creditData=pd.read_csv("credit_data.csv")

#print(creditData.head()) #shows the top 5 entries of datset
#print(creditData.describe()) #shows the most important details of the datset
#print(creditData.corr()) #show the correlation between different featues ina matrix form'

features=creditData[["duration_in_month","credit_amount"]]  #hence its a multivalus lgistic regression
targets=creditData.default  #this tells the final number of probablity if the payer would payback or not which here  is 0.7

feature_train,feature_test,target_train,target_test=train_test_split(features,targets,test_size=0.3)

model=RandomForestClassifier(n_estimators=1000,max_features='sqrt')
fitted_model=model.fit(feature_train,target_train)
predictions=fitted_model.predict(feature_test)

print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))