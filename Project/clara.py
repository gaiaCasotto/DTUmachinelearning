#summary statistics for report 1
import numpy as np
import preprocessing_lib as data
from scipy.stats import zscore
import pandas as pd
from matplotlib.pyplot import figure, subplot, savefig, hist, ylim, title, plot, legend, show,  xlabel, ylabel, xticks, yticks

#get data from data preprocessing file
[X, y, attributeNames] = data.get_data_matrix() #matrix of continous values
#[X_cont, attributeNames_cont] = data.get_cont_matrix(X, attributeNames)


not_cont_att = [1, 2, 5, 6, 8, 10]
X_cont = np.delete(X, not_cont_att, axis=1)
X_cont = X_cont.astype(float) #For some reason they are not seen as numbers
attributeNames_cont = np.delete(attributeNames, not_cont_att)


#Standardize the data:
X_standard = zscore(X_cont, ddof= 1)

data.data_analysis(X_standard, y, attributeNames_cont)

## Weird histogram for outliers
figure(figsize=(14,9))
m = [1, 2, 4] #RestingBP, Cholesterol and Oldpeak
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X_standard[:,m[i]],50)
    xlabel(attributeNames_cont[m[i]])
    #ylim(0, 400) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('Histogram (selected attributes)')
savefig('hist_selected_attrs.png')

## Scatter plot (matrix)
NumAtr = len(attributeNames_cont)
classNames = ['No heart disease', 'Heart disease']
figure(figsize=(12,12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(len(classNames)):
            class_mask = (y==c)
            plot(X_standard[class_mask, m2], X_standard[class_mask, m1], '.')
            if m1==NumAtr-1:
                xlabel(attributeNames_cont[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames_cont[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)
show()
savefig('images/scatterplot_matrix.png')

## Correlation matrix including y and binary attrs.
not_include_att = [2, 6, 10]
X_new = np.delete(X, not_include_att, axis=1)
X_new = X_new.astype(float) #For some reason they are not seen as numbers
attributeNames_new = np.delete(attributeNames, not_include_att)
savefig('images/corr_matrix_y_binary.png')


#Standardize the data:
X_std_new= zscore(X_new, ddof= 1)
y = y.reshape(-1, 1)

X_for_corr = np.concatenate((X_std_new,y), axis = 1)

correlation_matrix = np.corrcoef(X_for_corr, rowvar=False)

column_names = np.concatenate((attributeNames_new, ["Heart_Disease"]))
row_names = np.concatenate((attributeNames_new, ["Heart_Disease"]))

df_corr= pd.DataFrame(correlation_matrix, columns=column_names, index=row_names)



# TODO: Now that i have changed STLope do the PCA again
# encode the categorical values and find the correlation
