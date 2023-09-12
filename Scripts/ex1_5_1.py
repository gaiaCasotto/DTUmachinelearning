import numpy as np
import pandas as pd

# Load the Heart Disease csv data using the Pandas library
filename = '../heart.csv'
df = pd.read_csv(filename)

#eliminating the 0s in restingBP  //there is only ONE 0, so we delete it
for i in range(len(df['RestingBP'])):
    if df['RestingBP'].values[i] == 0:
        print(i)
        df = df.drop(i)
        break;
        
X = df.drop(columns=['HeartDisease']).values
y = df['HeartDisease'].values

# Replace 0 values in column 4 (chol) with NaN
X[X[:, 4] == 0, 4] = np.nan
print(X[:, 4])
print()
# Convert the column with NaN values to float
X[:, 4] = X[:, 4].astype(float)

# Impute NaN values with the median of the column
impute_val = np.nanmedian(X[:, 4])
print(f"imp val is {impute_val}")

for i in range(0, len(X[:, 4])):
    if np.isnan(X[i,4]):
        X[i,4] = impute_val

print(f"final {X[:, 4]}")

# We can extract the attribute names that came from the header of the csv
attributeNames = df.columns[:-1].values

N, M = X.shape



