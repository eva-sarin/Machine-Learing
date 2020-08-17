# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:18:57 2020

@author: Dell
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm #support vector machine

xBlue=np.array([0.3,0.5,1,1.4,1.7,2])
yBlue=np.array([1,4.5,2.3,1.9,8.9,4.1])

xRed=np.array([3.3,3.5,4,4.4,5.7,6])
yRed=np.array([7,1.5,6.3,1.9,2.9,7.1])

X=np.array([(0.3,1),[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4.4,1.9],[5.7,2.9],[6,7.1]])
Y=np.array([0,0,0,0,0,0,1,1,1,1,1,1])  #0 is blue class while 1 is red class

plt.plot(xBlue,yBlue,'ro',color='blue')
plt.plot(xRed,yRed,'ro',color='red')
plt.plot(6,6,'ro',color='green')

# Important parameters for SVC1: gamma and c
#gamma--> defines how far the influence of a single training example reaches
#low value:influencereaches far  High value:influence reaches close

#C --> trades off hyperplane surfacee simplicity + training eg misclassifications
# low value =simple/smooth hyperplane surface
#high value=all trainng eg classified correctly but complex surface

classifier=svm.SVC() #support vector classifier
classifier.fit(X,Y)

print(classifier.predict([6,6]))

plt.axis([-0.5,10,-0.5,10])
plt.show()