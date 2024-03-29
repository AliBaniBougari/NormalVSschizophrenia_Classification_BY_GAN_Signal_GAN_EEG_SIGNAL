import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import csv
from matplotlib import style
import string
from collections import Counter
import sys
import pickle
from sklearn.model_selection import train_test_split
import random

from mlxtend.plotting import plot_decision_regions
from sklearn import svm

clf = svm.SVC(kernel='rbf' , C=9,gamma=100)

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
print(y.shape)

X=np.reshape(X,(len(X),15*7680))

print(X.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
clf.fit(X_train,y_train)

print(X_test.shape)

print("Testing acc=",clf.score(X_test,y_test)*100)

print("Training acc=",clf.score(X_train,y_train)*100)
