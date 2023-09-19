#summary statistics for report 1
import numpy as np
import heart_project as data

#get data from data preprocessing file
X, y, attributeNames = data.get_data_matrix() #matrix of continous values
X_cont = data.get_cont_matrix(X)

print("SUMMARY STATISTICS:")
for i in range(0, len(X_cont[1,:])):  #first row, all columns
    col = X_cont[:, i]
    mean_col   = col.mean()
    std_col    = col.std(ddof=1)
    median_col = np.median(col)
    range_col  = col.max() - col.min()
    
    print(f"column {i}: mean = {mean_col}, std = {std_col}, median = {median_col}, range = {range_col}")

data.pca_analysis(X, y, attributeNames)

#matrix with corelation coefficients of columns
correlation_matrix = np.corrcoef(X_cont, rowvar=False)
