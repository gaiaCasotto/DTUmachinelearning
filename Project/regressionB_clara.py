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
Error_baseline = np.empty((K1,1))
Error_baseline = np.empty((K1,1))

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
    Error_baseline[k] = np.square(y_train_rlr-y_train_rlr.mean()).sum(axis=0)/y_train_rlr.shape[0]
    Error_baseline[k] = np.square(y_test_rlr-y_test_rlr.mean()).sum(axis=0)/y_test_rlr.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M_rlr)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train_rlr-X_train_rlr @ w_rlr[:,k]).sum(axis=0)/y_train_rlr.shape[0]
    Error_test_rlr[k] = np.square(y_test_rlr-X_test_rlr @ w_rlr[:,k]).sum(axis=0)/y_test_rlr.shape[0]
    
    opt_lambdas[k] = opt_lambda
    
    ##################################
    
    ########## ANN ################
    X_train_ann = X_ann[train_index]
    y_train_ann = y_ann[train_index]
    X_test_ann = X_ann[test_index]
    y_test_ann = y_ann[test_index] 
    
    errors_test = np.empty(len(n_hidden_units_values))
    for n in range(0, len(n_hidden_units_values)):
        n_hidden_units1 = n_hidden_units_values[n]
        n_hidden_units2 = n_hidden_units1
        errors_test[n] = get_ann(X_train_ann, y_train_ann, K2, n_hidden_units1, n_hidden_units2)
    
    opt_val_err = np.min(errors_test)
    opt_n_hidden_units[k] = n_hidden_units_values[np.argmin(errors_test)]
        
    Error_train_ann[k] = opt_val_err
    
    # # inner loop:
    # CV2 = model_selection.KFold(K2, shuffle=True)
    # errors_test = np.empty((K2,len(n_hidden_units_values)))
    # errors_train = np.empty((K2,len(n_hidden_units_values)))
    # # y_train_ann = y_train_ann.squeeze()
     
    # for (k2, (train_index, test_index)) in enumerate(CV2.split(X_train_ann,y_train_ann)):
    #     # Extract training and test set for current CV fold, convert to tensors
    #     X_train = torch.Tensor(X_train_ann[train_index,:])
    #     y_train = torch.Tensor(y_train_ann[train_index])
    #     X_test = torch.Tensor(X_train_ann[test_index,:])
    #     y_test = torch.Tensor(y_train_ann[test_index])
        
    #     for n in range(0, len(n_hidden_units_values)):
    #     # for n_hidden_units2 in range(1, n_hidden_units1+1):
    #         n_hidden_units1 = n_hidden_units_values[n]
    #         n_hidden_units2 = n_hidden_units1
    #         model = lambda: torch.nn.Sequential(
    #             torch.nn.Linear(M, n_hidden_units1),  # M features to n_hidden_units
    #             torch.nn.ReLU(),  # 1st transfer function
    #             torch.nn.Linear(n_hidden_units1, n_hidden_units2),  # Add another hidden layer with n_hidden_units units
    #             torch.nn.ReLU(),  # 2nd transfer function
    #             torch.nn.Linear(n_hidden_units2, 1),  # n_hidden_units to 1 output neuron
    #             # no final transfer function, i.e. "linear output"
    #         )
    #         loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    

    #         # Train the net on training data
    #         net, final_loss, learning_curve = train_neural_net(model,
    #                                                         loss_fn,
    #                                                         X=X_train,
    #                                                         y=y_train,
    #                                                         n_replicates=n_replicates,
    #                                                         max_iter=max_iter)
            
    #         # Determine estimated class labels for test set
    #         y_test_est = net(X_test)
    #         y_train_est = net(X_train)
            
    #         # Determine errors and errors
    #         se_test = (y_test_est.float()-y_test.float())**2 # squared error
    #         mse_test = (sum(se_test).type(torch.float)/len(y_test)).data.numpy() #mean
    #         errors_test[k2, n] = mse_test # store error rate for current CV fold 
        
    #         se_train = (y_train_est.float()-y_train.float())**2 # squared error
    #         mse_train = (sum(se_train).type(torch.float)/len(y_train)).data.numpy() #mean
    #         errors_train[k2, n] = mse_train # store error rate for current CV fold 
            

    #         opt_val_err = np.min(np.mean(errors_test,axis=0))
    #         opt_n_hidden_units[k] = n_hidden_units_values[np.argmin(np.mean(errors_test,axis=0))]
        
    #         Error_train_ann[k] = opt_val_err
    #         Error_test_ann[k] = np.mean(errors_test)   
    ###############################
    
    k+=1


print('Ran!')