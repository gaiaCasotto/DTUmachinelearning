import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import heart_project as data
from scipy.stats import chi2_contingency


path = '/home/codespace/DTUmachinelearning/Project/images'

#---DEFINE FUNCTION----> get categorical matrix
def get_cat_matrix(X):
    #not_cont_att = [1, 2, 5, 6, 8, 10]
    cont_att = [0,3,4,7,9]
    X_cat = np.delete(X, cont_att, axis = 1)
    return X_cat    

X, y, attribute_names = data.get_data_matrix()
X_cont, cont_attributes = data.get_cont_matrix(X, attribute_names) #matrix of continuous values

data.data_analysis(X_cont, y, cont_attributes)
data.pca_analysis(X, y, attribute_names)

print(y)
print(len(attribute_names))
not_cont_att = [1, 2, 5, 6, 8, 10]

#---SCATTER PLOTS FOR CONTINUOUS ATTRUBUTES----
'''cont_attributes = attribute_names
for index in sorted(not_cont_att, reverse=True):
    cont_attributes = np.delete(cont_attributes, index)
'''
a_len = len(cont_attributes)
print(a_len)
print(cont_attributes)

'''  THESE PLOTS ARE PRETTY MUCH USELESS... dont really know how to plot the continous variables
plt.figure(figsize=(12, 10))
for i in range(X_cont.shape[1]):
    plt.subplot(3, 2, i + 1)  # Adjust subplot layout as needed
    plt.scatter(X_cont[:, i], y, alpha=0.5)
    plt.xlabel(attribute_names[i])
    plt.ylabel('HeartDisease')
    plt.title(f'Scatter Plot: {attribute_names[i]} vs HeartDisease')

    plt.tight_layout()  # Adjust subplot spacing
    plt.show()
'''
'''for ind1 in range(a_len):
    col1 = X_cont[:, ind1]
    sns.scatterplot(x=col1, y=col1, hue = y, palette='viridis')
    #sns.lmplot(x=attribute_names[ind1], y= attribute_names[ind2], hue= "heartDisease", data=X_cont) #X_cont needs to be a df to work
    plt.title(f'Scatter Plot: {cont_attributes[ind1]} in correlation with hearDisease')
    plt.xlabel(cont_attributes[ind1])
    plt.ylabel('HeartDisease')
    plt.show()
    '''
 
#---correlation matrix for continuous attributes----
df_cont = pd.DataFrame(X_cont, columns=cont_attributes)
df_cont['HeartDisease'] = y

# Calculate the correlation matrix
correlation_matrix = df_cont.corr()

# Create a heatmap of the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Continuous Attributes with HeartDisease')
#full_path = path + 'CorrelationHeatmap.png'
full_path = 'images/CorrelationHeatmap.png'
plt.savefig(full_path)
#plt.show()

    
#----plots for the categorical values-----
X_cat   = get_cat_matrix(X)
cat_att = [0, 3, 4, 7, 9]

# Get categorical matrix and attributes
X_cat = get_cat_matrix(X)
cat_att = [0, 3, 4, 7, 9]
cat_attributes = [attribute_names[i] for i in range(len(attribute_names)) if i not in cat_att]


# Create a DataFrame from the categorical matrix
df_cat = pd.DataFrame(X_cat, columns=cat_attributes)
df_cat['HeartDisease'] = y

# SEX: 0 is female, 1 is male
# EX Angina: 0 is No, 1 is Yes

# Create a clustered bar plot
plt.figure(figsize=(10, 6))
for ind1 in range(a_len):
    sns.countplot(data=df_cat, x=cat_attributes[ind1], hue='HeartDisease', palette='Set1')
    plt.title('Clustered Bar Plot of Categorical Attributes')
    x_label = cat_attributes[ind1]
    plt.xlabel(cat_attributes[ind1])
    plt.ylabel('Count')
    plt.legend(title='', loc='upper right')
    L = plt.legend()
    L.get_texts()[0].set_text('has CVD')
    L.get_texts()[1].set_text('no CVD')
    plt.savefig('images/countPlot_' + cat_attributes[ind1] + '.png')
    #plt.show()


#-----correlation matrix for categorical attributes----
cat_attributes.append('HeartDisease')
chi2_stats = pd.DataFrame(index=cat_attributes, columns=cat_attributes)

# Calculate chi-squared statistics for each pair of categorical attributes
for attribute1 in cat_attributes:
    for attribute2 in cat_attributes:
        if attribute1 != attribute2:
            contingency_table = pd.crosstab(df_cat[attribute1], df_cat[attribute2])
            chi2, _, _, _ = chi2_contingency(contingency_table)
            chi2_stats.at[attribute1, attribute2] = chi2

# Create a heatmap of chi-squared statistics
plt.figure(figsize=(10, 8))
sns.heatmap(chi2_stats.astype(float), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Chi-Squared Correlation Heatmap of Categorical Attributes')
#plt.show()
plt.savefig('images/CorrelationHeatmapCategorical.png')

