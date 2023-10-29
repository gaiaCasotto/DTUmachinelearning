#Regression part A
import numpy as np
import heart_project as data
from scipy.stats import zscore
from toolbox_02450 import rlr_validate
from matplotlib.pyplot import figure, subplot, savefig, semilogx, loglog, title, grid, legend, show,  xlabel, ylabel, xticks, yticks

#get data from data preprocessing file
[X, y, attributeNames] = data.get_data_matrix() #matrix of continous value


# Add offset attribute
MaxHR_index = 7
X = np.delete(X, MaxHR_index, axis=1)
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
y = X[:, MaxHR_index]
attributeNames = ['Offset']+attributeNames
N, M = X.shape

# Values of lambda
lambdas = np.power(10.,range(-20,20))


# extract training and test set for current CV fold
internal_cross_validation = 10    

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, internal_cross_validation)

# Standardize X
X_std= zscore(X, ddof= 1)


Xty = X.T @ y
XtX = X.T @ X

# Estimate weights for the optimal value of lambda, on entire training set
lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()



# Display the results for the last cross-validation 
figure(1, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()