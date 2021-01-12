import sqlite3
import numpy as np
import pandas as pd
import re
import string
import nltk
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import collections
from sklearn import model_selection



con = sqlite3.connect('final.sqlite')
filtered_data = pd.read_sql_query("""
SELECT * FROM Reviews 
""",con)


x = filtered_data['cleanedText']
y = filtered_data['Score']

count_vect = CountVectorizer(ngram_range=(1,1))
xx = count_vect.fit_transform(x) #compute the bag of words


x_train, x_test, y_train, y_test = model_selection.train_test_split(xx,y, test_size=0.3, random_state=0) #split the data into two(_train and _test with _test data to be 30% of total data(test_size). Training data and Test data(unseen)
myrange = list(range(0,10))
neighbors = list(filter(lambda x: x % 2 != 0, myrange)) #odd number in myrange
cv_score = []
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)
    scores = cross_val_score(knn,x_train,y_train,cv=10,scoring='accuracy') #k-fold. k=10
    cv_score.append(scores)

MSE = [1-x for x in cv_score] #misclassification error
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is {}'.format(optimal_k))
#use optimal k to predict your unseen test and get the accuracy
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(x_train,y_train)
pred = knn_optimal.predict(x_test)
acc = accuracy_score(y_test,pred) * 100
print('\nThe accuracy of the knn classifier for k = {0} for an unseen data prediction is {1}'.format(optimal_k, acc))






