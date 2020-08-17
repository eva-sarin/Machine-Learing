# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:44:50 2020

@author: Dell
"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer()

tfidf=vectorizer.fit_transform(["I like machine learning and cllutering algorithms","Apples,oranges and any kind of frits are healthy","Is it feasible with machine learning algorithm?","my family is happy because of the healthy fruits"])

#print(tfidf.A) #cant be read by humans hence we use
print(((tfidf*tfidf.T).A))  #similarity matrix