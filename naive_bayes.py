# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:31:35 2020

@author: Dell
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("credit_data.csv")

#log reg accuracy was 93%
#with knn it was 97.5%
#otherwise 84%

features=creditData[["duration_in_month","credit_amount"]]  #hence its a multivalus lgistic regression
target=creditData.default  #this tells the final number of probablity if the payer would payback or not which here  is 0.7

feature_train,feature_test,target_train,target_test=train_test_split(features,target,test_size=0.3)

model=GaussianNB()
model.fit=model.fit(feature_train,target_train)
predictions=model.fit.predict(feature_test)

print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))