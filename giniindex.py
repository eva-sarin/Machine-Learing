# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 02:05:56 2020

@author: Dell
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

data=pd.read_csv("C:\\Users\\Dell\\Desktop\\ML and DL udemy\\irisdata.csv")

data.features=data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
data.targets=data.Class  #target variable

feature_train, feature_test, target_train, target_test=train_test_split(data.features,data.targets,test_size=.2)

model=DecisionTreeClassifier(criterion='gini') #various criterias are there in documentations of sklearn.tree
model.fitted=model.fit(feature_train,target_train)
model.predictions=model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test,model.predictions))