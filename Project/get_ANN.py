import preprocessing_lib as pplib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

def get_ann(X, y, K, n_hidden_units1, n_hidden_units2, n_replicates=1, max_iter = 12000):
    """
    Get a neural network for our specific problem, it is used for 
    testing the models in the inner loop.
    
    Our different models have different number of hidden units in each 
    layer
    
    """
    # K-fold crossvalidation
    CV = model_selection.KFold(K, shuffle=True)
    N, M = X.shape
    
    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units1),  # M features to n_hidden_units
        torch.nn.ReLU(),  # 1st transfer function
        torch.nn.Linear(n_hidden_units1, n_hidden_units2),  # Add another hidden layer with n_hidden_units units
        torch.nn.ReLU(),  # 2nd transfer function
        torch.nn.Linear(n_hidden_units2, 1),  # n_hidden_units to 1 output neuron
        # no final transfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    se_tot = []
    #print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train,
                                                        y=y_train,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
        
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_test_est = net(X_test)
        y_train_est = net(X_train)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean squared error
        errors.append(mse) # store error rate for current CV fold
        
        print("se of this inner fold: ", se)
        se_tot.append(se.detach().numpy().squeeze())
        print("se appended: ", se)
        
    se_tot = np.concatenate(se_tot)  
    print("final Se: ", se_tot)
    
    return np.mean(errors), se_tot