from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


z = [45,67,90,78,67,45,92,10,3,7]
x = np.array(z)
y = ['positive','negative','negative','negative','positive','positive','negative','negative','positive','negative']
f = np.array([65])
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x.reshape(-1, 1), y)
pre = knn.predict([[65]])
print(pre)