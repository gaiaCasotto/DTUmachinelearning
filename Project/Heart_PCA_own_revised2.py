# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 21:48:52 2023

@author: Eszter
"""

import numpy as np
import pandas as pd

raw_heart = pd.read_csv('../heart.csv', sep=';')
print(raw_heart.head)

#set all zeros in columns 3 and 4 to nan
cols_NA=raw_heart.iloc[:,3:5]
for col in cols_NA:
    raw_heart[col] = raw_heart[col].replace(0, np.nan)

#replace nan for mean

for col in cols_NA:
    raw_heart[col].fillna(raw_heart[col].mean(), inplace=True)

#Optional: Check means of columns, find our replaced values with the function abs that 
raw_heart.mean(0)
tolerance=1e-2
raw_heart[abs(raw_heart['RestingBP'] - 132.540894)<tolerance]
raw_heart[abs(raw_heart['Cholesterol'] - 244.635389)<tolerance]
#The values are in the dataframe
    
#drop non-numeric values from dataframe (leave sex)
heart = raw_heart.iloc[:, [0,1, 3, 4, 5, 7, 9]]

#transform sex to binary encoding, then one-out-of-K
sex_Labels= heart.iloc[:,1]
sex_Names= sorted(set(sex_Labels))
sex_Dict= dict(zip(sex_Names, range(len(sex_Labels))))
Sex_col= np.asarray([sex_Dict[value] for value in sex_Labels])
heart['Sex'] = Sex_col

Sex = np.array(heart.iloc[:,1],dtype=int).T #=Sex_col
K=Sex.max()+1
Sex_encoding=np.zeros((Sex.size,K))
Sex_encoding[np.arange(Sex.size),Sex]=1
#extract values from dataframe into an array
heart_array = heart.values
heart_array=np.concatenate((heart.iloc[:,[0,2,3,4,5,6]],Sex_encoding),axis=1)
#Sex_encoding are now columns 6=female and 7=male

#One-out-of-K encoding for Fasting BS
FastingBS = np.array(heart.iloc[:,4],dtype=int).T 
K2=FastingBS.max()+1
FastingBS_encoding=np.zeros((FastingBS.size,K))
FastingBS_encoding[np.arange(FastingBS.size),Sex]=1
heart_array_ready=np.concatenate((heart_array[:,[0,1,2,4,5,6,7]],FastingBS_encoding),axis=1)
#now in heart_array_ready columns 5=female, 6=male, 7=FastingBS 1, 8=FastingBS=0

#define Y as array
heart_y=raw_heart.iloc[:,-1]
Y=heart_y.values

#subtract mean value from data
X=heart_array_ready - np.ones((len(Y),1))*heart_array_ready.mean(axis=0)

#standardize to sd=1
X = X*(1/np.std(X,0))

#convert all to floats
X=X.astype(float)

#compute PCA
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, legend
from scipy.linalg import svd
U,S,V = svd(X,full_matrices = False)

#compute variance explained by principal components
rho = (S*S)/(S*S).sum()

#How many principal components are necessary to explain 90% of the variance?
p_components = 0
cumulative_variance= 0.0
for item in rho:
    cumulative_variance += item
    p_components += 1
    if cumulative_variance >= 0.9:
        break
    
print(f'Principle components needed to explain over 90% variance: {p_components}')
print(f'Explained variance: {cumulative_variance}')

#Plot the explained variance
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[0.9,0.9],'k--')
plt.title('Variance explained by the principle components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['individual', 'cumulative', 'threshold'])
plt.grid()
plt.show()

#project centered data on component space
Z = X @ V

#%%
#plot the first two principal components of the heart dataset
i=0
j=1
C= sorted(set(Y))
len(C)
f=figure()
title('Prediction of Heart Disease Dataset PCA')
for c in range(len(C)):
    class_mask=Y==c
    plot(Z[class_mask,i],Z[class_mask, j],'o',alpha=0.5)
legend(['Normal','Heart Disease'])
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
