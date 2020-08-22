# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 00:18:29 2020

@author: Dell
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data=pd.read_csv("C:\\Users\\Dell\\Desktop\\ML and DL udemy\\irisdata.csv")

data.features=data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
data.targets=data.Class  #target variable

#with grid search you can find an optimal parameter "parameter tuning"
param_grid={'max_depth':np.arange(1,10)}

#in every iteration data is spittered randomly in cross validation +decisionTreeClassifier
#initialize the tree randomly:thats why you get different results!!
tree=GridSearchCV(DecisionTreeClassifier(),param_grid)

feature_train,feature_test,target_train,target_test=train_test_split(data.features,data.targets,test_size=.2)

tree.fit(feature_train,target_train)
tree_predictions=tree.predict_proba(feature_test)[:,1]

print("Best parameter with Grid Search:",tree.best_params_)