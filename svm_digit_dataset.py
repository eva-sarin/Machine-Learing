# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 01:03:00 2020

@author: Dell
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import datasets, classifiers and performance metrics
from sklearn.metrics import accuracy_score
from sklearn import datasets,svm,metrics

#the digits dataset
digits=datasets.load_digits()

#print("Digits\n",digits)

images_and_labels=list(zip(digits.images,digits.target))

#for index,(image,label) in enumerate(images_and_labels[:6]):
 #   plt.subplot(2,3,index+1)
  #  plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
   # plt.title('Target:%i' %label)
   
 # to apply a classifier on this data, we need to flatten the image,to
#turn the data in a (samples,feature) matrix:

n_samples=len(digits.images)
data=digits.images.reshape((n_samples,-1))
#print("Data\n",data)

#Create a classifier =a support vector classifier
classifier=svm.SVC(gamma=0.001)

#we learn the digits n the first half of the digits
trainTestSplit=int(n_samples*0.75)
classifier.fit(data[:trainTestSplit],digits.target[:trainTestSplit])

#now predict the value of the digit on the second half
expected=digits.target[trainTestSplit:]
predicted=classifier.predict(data[trainTestSplit:])

#print("classification report for classifier %s:\n%s\n")
# % (classifier, metrics.classification_report(expected,predicted))

print("confusion matrix:\n%s" %metrics.confusion_matrix(expected,predicted))
print(accuracy_score(expected,predicted)) 

#lets test on last few images
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r,interpolation='nearest')
print("prediction for test image:",classifier.predict(data[-1].reshape(1,-1)))
plt.show() 