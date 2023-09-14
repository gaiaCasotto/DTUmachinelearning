# %% Data treatment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd


# Load the Heart Disease csv data using the Pandas library
#filename = 'C:\\Users\\clara\\Desktop\\ML_Exercises\\Project_1\\Data\\heart.csv'
filename = '../heart.csv'
df = pd.read_csv(filename)

#eliminating the 0s in restingBP  //there is only ONE 0, so we delete it
for i in range(len(df['RestingBP'])):
    if df['RestingBP'].values[i] == 0:
        df = df.drop(i)
        break; #There is only one
        
# For sex, Female = 0, Male = 1
sex_Labels = df['Sex'] 
sex_Names = np.unique(sex_Labels)
sex_Dict = dict(zip(sex_Names,range(len(sex_Labels))))

sex_column = np.array([sex_Dict[sex] for sex in sex_Labels])
df['Sex'] = sex_column

X = df.drop(columns=['HeartDisease']).values
y = df['HeartDisease'].values

# For sex, Female = 0, Male = 1
sex_Labels = df['Sex'] 
sex_Names = np.unique(sex_Labels)
sex_Dict = dict(zip(sex_Names,range(len(sex_Labels))))

sex_column = np.array([sex_Dict[sex] for sex in sex_Labels])
df['Sex'] = sex_column
# Replace 0 values in column 4 (chol) with NaN
X[X[:, 4] == 0, 4] = np.nan

# Convert the column with NaN values to float
X[:, 4] = X[:, 4].astype(float)

# Impute NaN values with the median of the column
impute_val = np.nanmedian(X[:, 4])

for i in range(0, len(X[:, 4])):
    if np.isnan(X[i,4]):
        X[i,4] = impute_val

# We can extract the attribute names that came from the header of the csv
attributeNames = df.columns[:-1].values

N, M = X.shape

# %% PCA SECTION --> AMount of variation explained as a function of 
# the number of PCA components included

# Get continuous variables (including the ones that are 0 or 1):
not_cont_att = [2, 6, 8, 10]
X_cont = np.delete(X, not_cont_att, axis=1)
X_cont = X_cont.astype(float) #For some reason they are not seen as numbers
attributeNames_cont = np.delete(attributeNames, not_cont_att)
# Subtract mean value from data
Xc = X_cont - np.ones((N,1))*X_cont.mean(axis=0) #mean along columns (attributes)
# Different scale, standarise with std
Xc = Xc*(1/np.std(Xc,0))

# PCA by computing SVD of Xc
U,S,Vh = svd(Xc,full_matrices=False)
V = Vh.T    
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.xticks(np.arange(1, len(rho) + 1, 1))
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# %% PCA SECTION --> Principal directions of considered PCA components
N,M = X_cont.shape
num_pcs = 4
pcs = np.arange(0,num_pcs) 
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .11
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames_cont)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()

# %% PCA SECTION --> Data projected onto cosnidered principal components
#Projection:
Z = Xc @ V # = U @ S 
# Plot PCA of the data
f = plt.figure()
plt.title('pixel vectors of handwr. digits projected on PCs')
for c in [0,1]:
    # select indices belonging to class c:
    class_mask = (y == c)
    plt.plot(Z[class_mask,0], Z[class_mask,1], 'o')
plt.legend(['No heart disease', 'Heart disease'])
plt.xlabel('PC1')
plt.ylabel('PC2')

# %%
# Determine the number of rows and columns for the subplots
num_rows = (num_pcs + 1) // 2  # Use integer division to ensure an even number of rows
num_cols = 2

# Create a single figure with subplots arranged in rows and columns
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))  # Adjust figsize as needed

# Set a common title for all subplots
fig.suptitle('Data projected onto Principal Components')

# Loop through each PC and create subplots
for pc in range(num_pcs):
    row = pc // num_cols  # Calculate the row index
    col = pc % num_cols  # Calculate the column index

    for c in [0, 1]:
        # Select indices belonging to class c:
        class_mask = (y == c)
        axs[row, col].plot(Z[class_mask, pc], 'o', label='No heart disease' if c == 0 else 'Heart disease')
    axs[row, col].set_xlabel(f'PC{pc+1}')
    axs[row, col].set_ylabel('Projection Value')
    axs[row, col].legend()
    axs[row, col].grid()

# Remove any empty subplots if num_pcs is odd
if num_pcs % 2 != 0:
    fig.delaxes(axs[num_rows - 1, num_cols - 1])

plt.tight_layout()  # Ensures subplots are neatly arranged
plt.show()


# %%