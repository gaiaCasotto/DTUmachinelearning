# %% Data treatment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
from scipy import stats


# Load the Heart Disease csv data using the Pandas library
def get_data_matrix():  #returns X, y, attributeNames
    filename = '../heart.csv'
    #filename = '/home/codespace/DTUmachinelearning/heart.csv'
    #filename = 'C:\\Users\\clara\\Desktop\\ML_Exercises\\Project_1\\Data\\heart.csv'
    df = pd.read_csv(filename)
    print(df.describe().T)
    print(round(df['Cholesterol'].corr(df['HeartDisease']),3))
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

    # For ExerciseAngina, No = 0, Yes = 1
    EA_Labels = df['ExerciseAngina']
    EA_Names = np.unique(EA_Labels)
    EA_Dict = dict(zip(EA_Names,range(len(EA_Labels))))
    EA_column = np.array([EA_Dict[EA] for EA in EA_Labels])
    df['ExerciseAngina'] = EA_column

    df_with_cvd = df[df['HeartDisease'] == 1]
    df_healthy  = df[df['HeartDisease'] == 0]
    X_with_cvd  = df_with_cvd.drop(columns=['HeartDisease']).values
    X_healthy   = df_healthy.drop(columns=['HeartDisease']).values
    print(X_healthy)
    print()
    print(X_with_cvd)

    X = df.drop(columns=['HeartDisease']).values #maybe we should invert these lines?
    y = df['HeartDisease'].values
    #print(f"printing y {y}")

    # Replace 0 values in column 4 (chol) with NaN
    X[X[:, 4] == 0, 4] = np.nan
    X_with_cvd[X_with_cvd[:, 4] == 0, 4] = np.nan
    X_healthy[X_healthy[:, 4] == 0, 4] = np.nan
    
    # Convert the column with NaN values to float
    X[:, 4] = X[:, 4].astype(float)
    X_with_cvd[:, 4] = X_with_cvd[:, 4].astype(float)
    X_healthy[:, 4] = X_healthy[:, 4].astype(float)
    # Impute NaN values with the median of the column
    healthy_val = np.nanmean(X_healthy[:, 4])
    cvd_val     = np.nanmean(X_with_cvd[:, 4])
    print(f"cvd: = {cvd_val}, healthy = {healthy_val}")
    for i in range(0, len(X[:, 4])):
        if np.isnan(X[i,4]):
            if y[i] == 1: #if the patient had cvd, then cvd_val
                X[i,4] = cvd_val
            else:
                X[i,4] = healthy_val

    # We can extract the attribute names that came from the header of the csv
    attributeNames = df.columns[:-1].values
    return X, y, attributeNames
    
def get_cont_matrix(matrix, attributeNames):  #returns a matrix and cont_attributes array
    not_cont_att = [1, 2, 5, 6, 8, 10]
    X_cont = np.delete(matrix, not_cont_att, axis=1)
    print(X_cont)
    X_cont = X_cont.astype(float) #For some reason they are not seen as numbers
    cont_attributes = attributeNames
    for index in sorted(not_cont_att, reverse=True):
        cont_attributes = np.delete(cont_attributes, index)
    return X_cont, cont_attributes


def pca_analysis(X, y, attributeNames):  #returns nothing
    N, M = X.shape
    # %% PCA SECTION --> AMount of variation explained as a function of
    # the number of PCA components included

    # Get variables for PCA (continuous and including the ones that are 0 or 1):
    not_pca_att = [2, 6, 10]
    X_pca = np.delete(X, not_pca_att, axis=1)
    X_pca = X_pca.astype(float) #For some reason they are not seen as numbers
    attributeNames_cont = np.delete(attributeNames, not_pca_att)
    # Subtract mean value from data
    Xc = X_pca - np.ones((N,1))*X_pca.mean(axis=0) #mean along columns (attributes)
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
    plt.savefig('images/pca_variance.png')

    # %% PCA SECTION --> Principal directions of considered PCA components
    N,M = X_pca.shape
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
    plt.savefig('images/PCA_directions.png')

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
    plt.savefig('images/PCA_projection2D')
    
    # Now in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Projection on first 3 PCs')

    # Plot PCA of the data in 3D
    for c in [0, 1]:
        # Select indices belonging to class c:
        class_mask = (y == c)
        ax.scatter(Z[class_mask, 0], Z[class_mask, 1], Z[class_mask, 2], marker='o')

    # Add labels and legend
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(['No heart disease', 'Heart disease'])

    # Adjust the viewing angle (elevation and azimuth angles)
    ax.view_init(elev=30, azim=100)
    plt.show()
    plt.savefig('images/PCA_projection3D.png')

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
    plt.savefig('images/PCA_projection_sub.png')


    # %% Outliers
    # Create box plots to identify outliers for each attribute
    # continuous variables

    not_cont_att = [1, 2, 5, 6, 8, 10]
    X_cont, attributeNames_cont = get_cont_matrix(X, attributeNames)
    # attributeNames_cont = np.delete(attributeNames, not_cont_att)
    print(X_cont)
    fig = plt.figure()
    boxplot = sns.boxplot(data=X_cont)  # Replace 'df' with your DataFrame
    plt.title("Box Plot for Outliers")
    boxplot.set_xticklabels(attributeNames_cont)
    plt.show()
    fig.savefig('images/outliers_boxplot.png')


    # %% Normal disttribution?
    # Create histograms for each attribute and do normality test
    j = 1
    normality_results = []
    for i in range(0, X_cont.shape[1]):
        column = X_cont[:, i]
        plt.subplot(2, 3, j)
        sns.histplot(column, kde=True)
        plt.title(f'Histogram for {attributeNames_cont[i]}')
        j += 1
        # Perform Shapiro-Wilk test on the current column
        statistic, p_value = stats.shapiro(column)
        normality_results.append((i, p_value))

    # Display histograms
    plt.tight_layout()
    plt.savefig('images/histograms.png')
    plt.show()

    # Display the results of normality test
    for column, p_value in normality_results:
        alpha = 0.05  # Significance level
        if p_value > alpha:
            print(f"{column}: p-value={p_value:.4f} (The data appears to be normally distributed)")
        else:
            print(f"{column}: p-value={p_value:.4f} (The data does not appear to be normally distributed)")

def data_analysis(X_cont, y, attributeNames_cont):
    """
    X_cont has to be a matrix with continuous atributes, 
    not even binary
    This shows the boxplot standarized
    This shows graphs to see if the distribution is normal
    """
    ## Outliers
    fig = plt.figure()
    boxplot = sns.boxplot(data=X_cont)  # Replace 'df' with your DataFrame
    plt.title("Box Plot for Outliers")
    boxplot.set_xticklabels(attributeNames_cont)
    plt.show()
    fig.savefig('images/outliers_boxplot.png')
    
     # %% Normal disttribution?
    # Create histograms for each attribute and do normality test
    j = 1
    normality_results = []
    for i in range(0, X_cont.shape[1]):
        column = X_cont[:, i]
        plt.subplot(2, 3, j)
        sns.histplot(column, kde=True)
        plt.title(f'Histogram for {attributeNames_cont[i]}')
        j += 1
        # Perform Shapiro-Wilk test on the current column
        statistic, p_value = stats.shapiro(column)
        normality_results.append((i, p_value))

    # Display histograms
    plt.tight_layout()
    plt.savefig('images/histograms.png')
    plt.show()

    # Display the results of normality test
    for column, p_value in normality_results:
        alpha = 0.05  # Significance level
        if p_value > alpha:
            print(f"{column}: p-value={p_value:.4f} (The data appears to be normally distributed)")
        else:
            print(f"{column}: p-value={p_value:.4f} (The data does not appear to be normally distributed)")
