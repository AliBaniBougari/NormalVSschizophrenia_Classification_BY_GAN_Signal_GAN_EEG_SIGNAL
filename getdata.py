import os
import pickle
import pyedflib
import numpy as np
from tqdm import tqdm
import os.path as path

'''
first for  gan must train on dataset 1
pickle dataset1
for model contraction two dataset 

'''
normalfiles=os.listdir('dataset1/normal/')
features=[]
"""
normal part

"""
######################################get_dataset_1
print('='*30+' load_normal_data '+'='*30+'\n')

with tqdm(total=len(normalfiles)) as pbar:
    for file in normalfiles:
        feats=[]
        filename="dataset1/normal/"+file
        f = open(filename, 'r')     
        x = f.readline()
        sublist=[]
        i=0
        while x:
            if i<=7679:                     
                sublist.append(float(x))
            else:
                i=0
                sublist=np.array(sublist)
                feats.append(sublist)
                sublist=[]
                sublist.append(float(x))
            i+=1
            x=f.readline()
        feats=np.array(feats)
        features.append(feats)
        pbar.update(1)

################################# get_dataset_2
#for use in model del """ and contraction two dataset


"""

illfiles=os.listdir('dataset2/')

with tqdm(total=len(illfiles)) as pbar:
    for file in illfiles:
        filename="dataset2/"+file
        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file
        if 'h' in file :
            for i in np.arange(n):
                feats = f.readSignal(i)
                sublist_1=[]
                sublist=[]
                p = 0 
                for i in range(len(feats)):
                    if p <15:
                        if i%7680 == 0 :
                            sublist = feats[:7680]
                            feats = feats[7680:]
                            sublist_1.append(sublist)
                            sublist =[]
                            p+=1
                    else:
                        break
                        #print(np.array(sublist_1).shape)
                features.append(sublist_1)
            pbar.update(1)


"""
features = np.array(features)
with open('normal_features', 'wb') as fp:
    pickle.dump(features, fp)

print('number of normal data : '+str(features.shape[0]))
print('normal shape : '+str(features.shape))
"""
ill part

"""
######################################get_dataset_1
print('='*30+' load_ill_data '+'='*30+'\n')
illfiles=os.listdir('dataset1/ill/')
features=[]
with tqdm(total=len(illfiles)) as pbar:
    for file in illfiles:
        feats=[]
        filename="dataset1/ill/"+file
        f = open(filename, 'r')     
        x = f.readline()
        sublist=[]
        i=0
        while x:
            if i<=7679:                     
                sublist.append(float(x))
            else:
                i=0
                sublist=np.array(sublist)
                feats.append(sublist)
                sublist=[]
                sublist.append(float(x))
            i+=1
            x=f.readline()
        feats=np.array(feats)
        features.append(feats)
        pbar.update(1)

        
################################# get_dataset_2
#for use in model del """ and contraction two dataset

"""

illfiles=os.listdir('dataset2/')

with tqdm(total=len(illfiles)) as pbar:
    for file in illfiles:
        filename="dataset2/"+file
        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file
        if 's' in file :
            for i in np.arange(n):
                feats = f.readSignal(i)
                sublist_1=[]
                sublist=[]
                p = 0 
                for i in range(len(feats)):
                    if p <15:
                        if i%7680 == 0 :
                            sublist = feats[:7680]
                            feats = feats[7680:]
                            sublist_1.append(sublist)
                            sublist =[]
                            p+=1
                    else:
                        break
                features.append(sublist_1)
            pbar.update(1)



features=np.array(features)
print(features.shape)
"""
features = np.array(features)

with open('ill_features', 'wb') as fp:
    pickle.dump(features, fp)
print('number of ill data : '+str(features.shape[0]))
print('ill shape : '+str(features.shape))


