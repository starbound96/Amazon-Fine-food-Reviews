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


#final_counts = count_vect.fit_transform(filtered_data['cleanedText'].values)  #compute the bag of words
#filtered_data['vector']= final_counts
x = filtered_data['cleanedText']
y = filtered_data['Score']
vect = TfidfVectorizer(ngram_range=(1,1))
x_data = vect.fit_transform(x)#Computing the tfid
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data,y, test_size=0.3, random_state=0) #split the data into two(_train and _test with _test data to be 30% of total data(test_size). Training data and Test data(unseen)
x_train1, x_cvs, y_train1, y_cvs  = model_selection.train_test_split(x_train,y_train, test_size=0.3)#split my 70% data for training which is in x_train and y_train into two with x_cvs and y_cvs to be 30% of the training data. they will be cross validation data
acc_score = []
acc_no = []
for i in range(1,30,4): #instantiate learning model from k of 1 to 30 skipping even numbers due to majority vote selection
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train1, y_train1) #training my knn with my crossvalidation train
    pred = knn.predict(x_cvs) #using crossvalidation test data to predict in other to determine thr accuracy of my knn
    acc = accuracy_score(y_cvs,pred,normalize=True) * float(100)
    acc_score.append(acc)
    acc_no.append(i)
    print('\nCV accuracy for k = {0} is {1}'.format(i,acc))

k_index = acc_score.index(max(acc_score))
optimal_k = acc_no[k_index]
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(x_train, y_train)
pred_optimal = knn_optimal.predict(x_test)
accuracy = accuracy_score(y_test, pred_optimal) * float(100)
