import preprocessing_lib as pplib
from get_ANN import get_ann
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from toolbox_02450 import rlr_validate
import scipy.stats as st


#load data
X, y, attributeNames = pplib.get_data_matrix()
X, y, attributeNames = pplib.change_y(X, y, attributeNames, 'HeartDisease', 'MaxHR') #with maxHR, training/test error jump from 0.12 to over 400 for some reason??????
N, M = X.shape

## RLR:
# Add offset attribute
X_rlr = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames_rlr = np.concatenate((['Offset'], attributeNames))
M_rlr = M+1 

## ANN:
attributeNames_ann = attributeNames.tolist()
X_ann = stats.zscore(X)
y_ann = np.expand_dims(y, axis=1)

## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 5
K2 = 5
CV = model_selection.KFold(K1, shuffle=True)

## Baseline
Error_baseline_train = np.empty((K1,1))
Error_baseline_test = np.empty((K1,1))

## RLR
lambdas = np.logspace(1, 3, 100)
mu = np.empty((K1, M_rlr-1))
sigma = np.empty((K1, M_rlr-1))
w_rlr = np.empty((M_rlr,K1))
Error_train_rlr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))
opt_lambdas = np.empty((K1,1))

## ANN
n_hidden_units_values = np.arange(3, 7) 
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 13000
Error_train_ann = np.empty((K1,1))
Error_test_ann = np.empty((K1,1))
opt_n_hidden_units = np.empty((K1,1))

## SETUP I
z_baseline = np.array([])
z_rlr = np.array([])
z_ann = np.array([])

k=0
for train_index, test_index in CV.split(X,y):
    ######### RLL and Baseline ########
    # extract training and test set for current CV fold
    X_train_rlr = X_rlr[train_index]
    y_train_rlr = y[train_index]
    X_test_rlr = X_rlr[test_index]
    y_test_rlr = y[test_index]  
    
    # internal loop of rlr
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_rlr, y_train_rlr, lambdas, K2)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train_rlr[:, 1:], 0)
    sigma[k, :] = np.std(X_train_rlr[:, 1:], 0)
    
    X_train_rlr[:, 1:] = (X_train_rlr[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test_rlr[:, 1:] = (X_test_rlr[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train_rlr.T @ y_train_rlr
    XtX = X_train_rlr.T @ X_train_rlr
    
    # Compute mean squared error without using the input data at all
    Error_baseline_train[k] = np.square(y_train_rlr-y_train_rlr.mean()).sum(axis=0)/y_train_rlr.shape[0]
    Error_baseline_test[k] = np.square(y_test_rlr-y_test_rlr.mean()).sum(axis=0)/y_test_rlr.shape[0]

    z_baseline = np.concatenate([z_baseline, np.square(y_test_rlr-y_test_rlr.mean())])
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M_rlr)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train_rlr-X_train_rlr @ w_rlr[:,k]).sum(axis=0)/y_train_rlr.shape[0]
    Error_test_rlr[k] = np.square(y_test_rlr-X_test_rlr @ w_rlr[:,k]).sum(axis=0)/y_test_rlr.shape[0]
    
    z_rlr = np.concatenate([z_rlr, np.square(y_test_rlr-X_test_rlr @ w_rlr[:,k])])
    
    opt_lambdas[k] = opt_lambda
    
    ##################################
    
    ########## ANN ################
    X_train_ann = X_ann[train_index]
    y_train_ann = y_ann[train_index]
    X_test_ann = X_ann[test_index]
    y_test_ann = y_ann[test_index] 
    
    errors_test = np.empty(len(n_hidden_units_values))
    all_nets = []
    all_z_ann  = np.zeros((len(test_index), len(n_hidden_units_values)))
    for n in range(0, len(n_hidden_units_values)):
        n_hidden_units1 = n_hidden_units_values[n]
        n_hidden_units2 = n_hidden_units1
        ## inner loop:
        #errors_test[n], all_z_ann[:, n] = get_ann(X_train_ann, y_train_ann, K2, n_hidden_units1, n_hidden_units2)
        errors_test[n], net = get_ann(X_train_ann, y_train_ann, K2, n_hidden_units1, n_hidden_units2)
        all_nets.append(net)
    
    opt_index = np.argmin(errors_test)
    opt_net = all_nets[opt_index]
    
    X_train_ann_tensor = torch.Tensor(X_train_ann)
    y_train_ann_tensor = torch.Tensor(y_train_ann)
    X_test_ann_tensor = torch.Tensor(X_test_ann)
    y_test_ann_tensor = torch.Tensor(y_test_ann)
    
    y_test_ann_est = opt_net(X_test_ann_tensor)
    y_train_ann_est = opt_net(X_train_ann_tensor)
    #opt_val_err = np.min(errors_test)
    opt_n_hidden_units[k] = n_hidden_units_values[opt_index]

    # Determine errors and errors
    se_test = (y_test_ann_est.float()-y_test_ann_tensor.float())**2 # squared error
    mse_test = (sum(se_test).type(torch.float)/len(y_test_ann_tensor)).data.numpy() #mean squared error
    # errors.append(mse) # store error rate for current CV fold
    Error_test_ann[k] = mse_test
    
    z_ann = np.concatenate([z_ann, se_test.detach().numpy().squeeze()])
    
    se_train = (y_train_ann_est.float()-y_train_ann_tensor.float())**2 # squared error
    mse_train = (sum(se_train).type(torch.float)/len(y_train_ann_tensor)).data.numpy() #mean squared error
    # errors.append(mse) # store error rate for current CV fold
    Error_train_ann[k] = mse_train
    
    #z_ann = np.concatenate([z_ann, all_z_ann[:, np.argmin(errors_test)]])
    
    
    
    
    ###############################
    
    k+=1
    
### Confidence intervals and p-value
alpha = 0.05

z_RLR_vs_BASELINE = -(z_rlr-z_baseline)
CI_RLR_vs_BASELINE = st.t.interval(1 - alpha, len(z_RLR_vs_BASELINE) - 1, loc=np.mean(z_RLR_vs_BASELINE), scale=st.sem(z_RLR_vs_BASELINE))  # Confidence interval
p_RLR_vs_BASELINE = 2*st.t.cdf(-np.abs(np.mean(z_RLR_vs_BASELINE)) / st.sem(z_RLR_vs_BASELINE), df=len(z_RLR_vs_BASELINE) - 1)  # p-value

z_ANN_vs_BASELINE = -(z_ann-z_baseline)
CI_ANN_vs_BASELINE = st.t.interval(1 - alpha, len(z_ANN_vs_BASELINE) - 1, loc=np.mean(z_ANN_vs_BASELINE), scale=st.sem(z_ANN_vs_BASELINE))  # Confidence interval
p_ANN_vs_BASELINE = 2*st.t.cdf(-np.abs(np.mean(z_ANN_vs_BASELINE)) / st.sem(z_ANN_vs_BASELINE), df=len(z_ANN_vs_BASELINE) - 1)  # p-value

z_RLR_vs_ANN = -(z_rlr-z_ann)
CI_RLR_vs_ANN = st.t.interval(1 - alpha, len(z_RLR_vs_ANN) - 1, loc=np.mean(z_RLR_vs_ANN), scale=st.sem(z_RLR_vs_ANN))  # Confidence interval
p_RLR_vs_ANN = 2*st.t.cdf(-np.abs(np.mean(z_RLR_vs_ANN)) / st.sem(z_RLR_vs_ANN), df=len(z_RLR_vs_ANN) - 1)  # p-value

ci_Baseline = st.t.interval(1-alpha, df=len(z_baseline)-1, loc=np.mean(z_baseline), scale=st.sem(z_baseline))  # Confidence interval
ci_rlr = st.t.interval(1-alpha, df=len(z_rlr)-1, loc=np.mean(z_rlr), scale=st.sem(z_rlr))  # Confidence interval
ci_ann = st.t.interval(1-alpha, df=len(z_ann)-1, loc=np.mean(z_ann), scale=st.sem(z_ann))  # Confidence interval 

print("Individual confidence intervals: \n", ci_Baseline, 
      "\n", ci_rlr, "\n", ci_ann)

print("Confidence intervals: \n", CI_ANN_vs_BASELINE, 
      "\n", CI_RLR_vs_BASELINE, "\n", CI_RLR_vs_ANN)

print("p-values: \n", p_ANN_vs_BASELINE, 
      "\n", p_RLR_vs_BASELINE, "\n", p_RLR_vs_ANN)

for i in range(0, K1):
    print(Error_baseline_test[i], Error_test_rlr[i], Error_test_ann[i])


print('Ran!')