import numpy as np
import os
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

#find best k in knn 

def best_k():
    k = 0
    t_ac = 0 
    test_ac = 0
    print('start')
    pbar = tqdm(total = 100)
    for  i in range(1,101):
        neigh = KNeighborsClassifier(n_neighbors=1,p = 1)
        neigh.fit(X_train,y_train)
        if neigh.score(X_train,y_train)*100 > 90:
            if t_ac < neigh.score(X_train,y_train)*100:
                t_ac = neigh.score(X_train,y_train)*100
            if test_ac < neigh.score(X_test,y_test)*100:
                test_ac = neigh.score(X_test,y_test)*100
                k = i
        pbar.update(1)
    print("="*40)
    print('\n')
    print('best n_neighbors for this data ')
    print('n_neighbors : '+str(k))
    print(' in this n_neighbors train acc is : '+str(t_ac))
    print(' in this n_neighbors test acc is : '+str(test_ac))
    print('\n')
    print("="*40)
    return k 
#load data 

with open('normal_features', 'rb') as fp:
    normal_features=pickle.load(fp)



with open('ill_features', 'rb') as fp:
    ill_features=pickle.load(fp)
X=[]
y=[]
for f in normal_features:
    X.append(f)
for f in ill_features:
    X.append(f)
X=np.array(X)
print(X.shape)
for l in range(normal_features.shape[0]):
    y.append(1)
for l in range(ill_features.shape[0]):
    y.append(0)
y=np.array(y)
X=np.reshape(X,(len(X),15*7680))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
k = best_k()
neigh = KNeighborsClassifier(n_neighbors=k,p = 1)
neigh.fit(X_train,y_train)
print(X_test.shape)
print("Testing acc=",neigh.score(X_test,y_test)*100)
print("Training acc=",neigh.score(X_train,y_train)*100)
input('pres any keys for exit ')
