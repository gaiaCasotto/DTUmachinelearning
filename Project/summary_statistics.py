#summary statistics for report 1
import numpy as np

#get data....
x = [] #matrix of continous values

for i in len(x[1,:]):  #first row, all columns
    col = x[:, 1]
    mean_col   = col.mean()
    std_col    = col.std(ddof=1)
    median_col = np.median(col)
    range_col  = col.max() - col.min()
