#summary statistics for report 1
import numpy as np
import heart_project as data
from scipy.stats import zscore

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