#to select the optimal value of the regularization parameter λ
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
import numpy as np
import preprocessing_lib as pplib
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

#load data
X, y, attribute_names = pplib.get_data_matrix()
X, y, attribute_names = pplib.change_y(X, y, attribute_names, 'HeartDisease', 'MaxHR') #with maxHR, training/test error jump from 0.12 to over 400 for some reason??????
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attribute_names = np.concatenate((['Offset'], attribute_names))
M = M+1  #WHY????


###########

cvf = 10 # K folds
# lambdas = np.power(10.,range(-5,9))
lambdas = np.logspace(0, 4, 100)
print(f"lambdas: {len(lambdas)}")

CV = model_selection.KFold(cvf, shuffle=True)
M = X.shape[1]
w = np.empty((M,cvf,len(lambdas)))
train_error = np.empty((cvf,len(lambdas)))
test_error = np.empty((cvf,len(lambdas)))

Error_train_nofeatures = np.empty((cvf,1))
Error_test_nofeatures = np.empty((cvf,1))
Error_train = np.empty((cvf,1))
Error_test = np.empty((cvf,1))

f = 0
y = y.squeeze()
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
        test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)

        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[f] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[f] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

        m = lm.LinearRegression().fit(X_train, y_train)
        Error_train[f] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        y_train_predict = m.predict(X_train)
        Error_test[f] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        y_test_predict = m.predict(X_test)
    
    f=f+1

# TODO: keep the w of the best lambda!
 
Error_test_rlr = np.min(np.mean(test_error,axis=0)) # Error_test_rlr
Error_train_rlr = np.min(np.mean(train_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))


opt_lambda_ix = np.where(lambdas==opt_lambda)[0]
w_opt_lambda = w[:, :, opt_lambda_ix].squeeze()


print(f"opt_lambda is {opt_lambda}")

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

show()

#show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr))
print('- Test error:     {0}'.format(Error_test_rlr))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr)/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr)/Error_test_nofeatures.sum()))

print('Weights:')
for m in range(M):
    print('{:>15} {:>15}'.format(attribute_names[m], np.round(w_opt_lambda[m,-1],2)))

