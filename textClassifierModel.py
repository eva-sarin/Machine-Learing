# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 00:09:55 2020

@author: Dell
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
trainingData=fetch_20newsgroups(subset='train',categories=categories, shuffle=True)

print ("\n".join(trainingData.data[0].split("\n")[:10]))
print ("target is:",trainingData.target_names[trainingData.target[0]])

#we just count the word occurences
countVectorizer=CountVectorizer()
xTrainCounts=countVectorizer.fit_transform(trainingData.data)
#print countVectorizer.vocabulary.get(u'software')

#we tranform the word occurences into tfidf
#TfudVectorizer=CountVectorizer + TfidfTransformer
tfidfTransformer=TfidfTransformer()
xTrainTfidf=tfidfTransformer.fit_transform(xTrainCounts)

model=MultinomialNB().fit(xTrainTfidf,trainingData.target)

new=['This has nothing to do with church or religion','software engineeering is getting hotter and hotter nowadays']
xNewCounts=countVectorizer.transform(new)
xNewTfidf=tfidfTransformer.transform(xNewCounts)

predicted=model.predict(xNewTfidf)

for doc,category in zip(new,predicted):
    print('%r-------->%s' % (doc,trainingData.target_names[category]))
    



