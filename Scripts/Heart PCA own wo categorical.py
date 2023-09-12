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

#replace nan with mean

for col in cols_NA:
    raw_heart[col].fillna(raw_heart[col].mean(), inplace=True)

#Optional: Check means of columns, find our replaced values 
raw_heart.mean(0)
tolerance=1e-2
raw_heart[abs(raw_heart['RestingBP'] - 132.540894)<tolerance]
raw_heart[abs(raw_heart['Cholesterol'] - 244.635389)<tolerance]
#The values are in the dataframe
    
#drop non-numeric values from dataframe
heart = raw_heart.iloc[:, [0, 3, 4, 7, 9]]


#extract values from dataframe into an array
heart_array = heart.values

#define Y as array 
heart_y=raw_heart.iloc[:,-1]
Y=heart_y.values

#subtract mean value from data
X=heart_array - np.ones((len(Y),1))*heart_array.mean(axis=0)


#standardize to sd=1
X = X*(1/np.std(X,0))


#convert all to floats
X=X.astype(float)

#compute PCA
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
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
