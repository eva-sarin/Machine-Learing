# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:48:25 2020

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

#read .csv file into dataframe usinf pandas
dataset= pd.read_csv('house_prices.csv')
size=dataset['LotArea']
price=dataset['MiscVal']



#machine learning handle arrays not dataframes
x=np.array(size).reshape(-1,1)
y=np.array(price).reshape(-1,1)

print(x)

#we use linear regression +firt() in the traiing
model=LinearRegression()
model.fit(x,y)

#MSE and Rvalue
regression_model_mse=mean_squared_error(x,y)
print("R squared value:", model.score(x,y))

#we get the b value after the model fit
#this is b0
print(model.coef_[0])
#this is b1
print(model.intercept_[0])

#visualize the dataset with the fitted model
plt.scatter(x,y,color='green')
plt.plot(x,model.predict(x),color='black')
plt.title("linear Regression")
plt.xlabel("size")
plt.ylabel("price")
plt.show()

#predicitng the rpoces 
print("prediction by the model:",model.predict(([2000])))