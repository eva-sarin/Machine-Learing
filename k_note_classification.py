# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 03:21:19 2020

@author: Dell
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

creditData=pd.read_csv("credit_data.csv")

features=creditData[["duration_in_month","credit_amount"]]  #hence its a multivalus lgistic regression
target=creditData.default  #this tells the final number of probablity if the payer would payback or not which here  is 0.7

model=LogisticRegression()
predicted=cross_validation.cross_val_predict(model,features,target,cv=10)

print(accuracy_score(target,predicted))