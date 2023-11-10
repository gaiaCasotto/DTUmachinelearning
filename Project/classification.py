#classification file
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from mlxtend.classifier import StackingClassifier

import preprocessing_lib as pplib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.io import loadmat
from toolbox_02450 import rlr_validate
from toolbox_02450 import mcnemar as tlbx_mcnemar


#BASELINE MODEL: The baseline will be a model which compute the largest class on the training data, and predict everything in the test-data as belonging to that class

def baseline_model(X_test_rows, y_train):
    count_zeros = 0
    count_ones  = 0
    for value in y_train:
        if value == 0:
            count_zeros += 1
        else:
            count_ones += 1
    cat = 0
    if count_ones > count_zeros:
        cat = 1
    y_pred = []
    for row in range(X_test_rows):
        y_pred.append(cat)
    return y_pred


sourceFile = open('results.txt', 'w')
X, y, attribute_names = pplib.get_data_matrix()
# configure the cross-validation procedure
K = 5
CV = KFold(n_splits=K, shuffle=True, random_state = 1)# enumerate splits
outer_results = list()
nns_pred_vector, nns_acc = [], []
lrs_pred_vector, lrs_acc = [], []
bls_pred_vector, bls_acc = [], []

nns_predictions, lrs_predictions, bls_predictions = [], [], []

trained_lrs = []
big_y_test  = []

k = 0
for train_ix, test_ix in CV.split(X, y):
    print(f"OUTER K = {k}")
    print(f"OUTER K = {k}", file = sourceFile)
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    # configure the cross-validation procedure
    CV_inner = CV
    model = None
    space = dict()
    
    print("ANN")
    model = MLPClassifier(max_iter = 500)
    space['max_iter'] = [400, 500]
    space['hidden_layer_sizes'] =  [(50), (100), (50, 50), (50, 100), (100, 50), (150), (75)] #search space
    search = GridSearchCV(model, space, scoring='accuracy', cv=CV_inner, refit=True)
    result = search.fit(X_train, y_train)
    #nns.append(result)
            # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
            # evaluate model on the hold out dataset
    yhat_nn = best_model.predict(X_test)
    for i in range(len(y_test)):
        nns_predictions.append(yhat_nn[i])
        if yhat_nn[i] == y_test[i]:
            nns_pred_vector.append(1) #correct prediction
        else:
            nns_pred_vector.append(0)  #wrong prediction
    sum = 0
    for p in nns_pred_vector:
        if p == 0:
            sum += 1
    E = sum / len(y_test)
    # evaluate the model
    acc = accuracy_score(y_test, yhat_nn)
    nns_acc.append(acc)
            # store the result
    outer_results.append(acc)
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_), file = sourceFile)
    print('NN error rate: %.3f' % (E), file = sourceFile)
    print("LR")
    space = dict()
    model = LogisticRegression(max_iter = 5000)
    space['C'] = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
    search = GridSearchCV(model, space, scoring='accuracy', cv=CV_inner, refit=True)
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    trained_lrs.append(best_model)  #saving them for the last part of classification
    # evaluate model on the hold out dataset
    yhat_lr = best_model.predict(X_test)
    for i in range(len(y_test)):
        lrs_predictions.append(yhat_lr[i])
        if yhat_lr[i] == y_test[i]:
            lrs_pred_vector.append(1) #correct prediction
        else:
            lrs_pred_vector.append(0) #wrong prediction
    # evaluate the model
    acc = accuracy_score(y_test, yhat_lr)
    sum = 0
    for p in lrs_pred_vector:
        if p == 0:
            sum += 1
    E = sum / len(y_test)
    lrs_acc.append(acc)
    # store the result
    outer_results.append(acc)
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_), file = sourceFile)
    print('LR error rate: %.3f' % (E), file = sourceFile)
    
    
    baseline_y_pred = baseline_model(X_test.shape[0], y_train)
    
    print("Baseline")
    for i in range(len(y_test)):
        bls_predictions.append(baseline_y_pred[i])
        if baseline_y_pred[i] == y_test[i]:
            bls_pred_vector.append(1) #correct prediction
        else:
            bls_pred_vector.append(0)  #wrong prediction
    #bls_predictions.append(bls_pred_vector)
    acc = accuracy_score(y_test, baseline_y_pred)
    sum = 0
    for p in bls_pred_vector:
        if p == 0:
            sum += 1
    E = sum / len(y_test)
    bls_acc.append(acc)
    print('>acc=%.3f, est= baseline ' % (acc), file = sourceFile)
    print('BL error rate: %.3f' % (E), file = sourceFile)

    outer_results.append(acc)
    for t in y_test:
        big_y_test.append(t)
    k += 1

#print(nns_acc)

[thetahat, CI, p] = tlbx_mcnemar(y_test, yhat_nn, yhat_lr, alpha = 0.05)
print(f'nn vs lr : {[thetahat, CI, p]} ', file = sourceFile)
[thetahat, CI, p] = tlbx_mcnemar(y_test, yhat_nn, baseline_y_pred, alpha = 0.05)
print(f'nn vs bl : {[thetahat, CI, p]} ', file = sourceFile)
[thetahat, CI, p] = tlbx_mcnemar(y_test, yhat_lr, baseline_y_pred, alpha = 0.05)
print(f'lr vs bl  : {[thetahat, CI, p]} ', file = sourceFile)
    
print('Mean accuracy of nn: %.3f (%.3f)' % (mean(nns_acc), std(nns_acc)), file = sourceFile)
print('Mean accuracy of lr: %.3f (%.3f)' % (mean(lrs_acc), std(lrs_acc)), file = sourceFile)
print('Mean accuracy of bl: %.3f (%.3f)' % (mean(bls_acc), std(bls_acc)), file = sourceFile)

'''
#mcNamaras testing
from scipy.stats import chi2_contingency, chi2
from statsmodels.stats.contingency_tables import mcnemar


def confidence_interval(chi2_stat):
    alpha = 0.05
    df = 2 # Degrees of freedom for the chi-square distribution == number of categories in the contingency tables

    

    print(f"Confidence Interval: ({lower_bound}, {upper_bound})")


def mcNemarTest(arr_nn, arr_lr, arr_bl):
# Create contingency tables
    c11, c12, c21, c22 = 0, 0, 0, 0
    for i in range(len(arr_nn)):
        if arr_nn[i] == 1:
            if arr_lr[i] == 1:
                c11 += 1
            else:
                c12 +=1
        else:
            if arr_lr[i] == 1:
                c21 +=1
            else:
                c22 += 1
    contingency_nn_lr = [
                    [c11,c12],
                    [c21,c22]
                    ]
    c11, c12, c21, c22 = 0, 0, 0, 0
    i = 0
    for i in range(len(arr_nn)):
        if arr_nn[i] == 1:
            if arr_bl[i] == 1:
                c11 += 1
            else:
                c12 +=1
        else:
            if arr_bl[i] == 1:
                c21 +=1
            else:
                c22 += 1
    contingency_nn_baseline = [
                        [c11,c12],
                        [c21,c22]
                        ]
    c11, c12, c21, c22 = 0, 0, 0, 0
    i = 0
    for i in range(len(arr_lr)):
        if arr_lr[i] == 1:  #if lr is correct
            if arr_bl[i] == 1: #and bl is correct
                c11 += 1
            else:    #if bl is wrong but lr is correct
                c12 +=1
        else:   #if lr is incorrect
            if arr_bl[i] == 1: #if bl is correct
                c21 +=1
            else:    #if bl is incorrect
                c22 += 1
    contingency_lr_baseline = [
                        [c11,c12],
                        [c21,c22]
                        ]

    # Perform McNemar's test for each pair
    #chi2_stat_nn_lr, p_value_nn_lr, _, _       = chi2_contingency(contingency_nn_lr)
    #chi2_stat_nn_bl, p_value_nn_baseline, _, _ = chi2_contingency(contingency_nn_baseline)
    #chi2_stat_lr_bl, p_value_lr_baseline, _, _ = chi2_contingency(contingency_lr_baseline)
    
    
    nn_lr    = mcnemar(contingency_nn_lr, exact = False)
    p_value_nn_lr, stat_nn_lr = nn_lr.pvalue, nn_lr.statistic
    nn_bl    = mcnemar(contingency_nn_baseline, exact = False)
    p_value_nn_bl, stat_nn_bl = nn_bl.pvalue, nn_bl.statistic
    lr_bl    = mcnemar(contingency_lr_baseline, exact = False)
    p_value_lr_bl, stat_lr_bl = lr_bl.pvalue, lr_bl.statistic
    
    print(f"stats: {stat_nn_lr} {stat_lr_bl} {stat_nn_bl}")
        
    print(f"mcnemar for nn vs lr: {mcnemar(contingency_nn_lr, exact = False)}")
    print(f"mcnemar for nn vs bl: {mcnemar(contingency_nn_baseline, exact = False)}")
    print(f"mcnemar for lr vs bl: {mcnemar(contingency_lr_baseline, exact = False)}")

    confidence_interval(stat_nn_lr)
    confidence_interval(stat_nn_bl)
    confidence_interval(stat_lr_bl)



for fold in range(K):
    print(f"fold : {fold}")
    mcNemarTest(nns_predictions[fold], lrs_predictions[fold], bls_predictions[fold])
    #in mcNemar's test, model A is better than model B if n12 > n21
'''
'''P-value for NN vs. LR (p_value_nn_lr):

If p_value_nn_lr < alpha: You would conclude that there is a significant difference between the Neural Network and Logistic Regression models.
If p_value_nn_lr ≥ alpha: You would not conclude a significant difference between the Neural Network and Logistic Regression models.


P-value for NN vs. Baseline (p_value_nn_baseline):
If p_value_nn_baseline < alpha: You would conclude that there is a significant difference between the Neural Network and Baseline models.
If p_value_nn_baseline ≥ alpha: You would not conclude a significant difference between the Neural Network and Baseline models.

P-value for LR vs. Baseline (p_value_lr_baseline):
If p_value_lr_baseline < alpha: You would conclude that there is a significant difference between the Logistic Regression and Baseline models.
If p_value_lr_baseline ≥ alpha: You would not conclude a significant difference between the Logistic Regression and Baseline models.

Remember that the choice of the significance level (alpha) is somewhat arbitrary and should be determined based on the context of your study and the level of confidence you require. Commonly used values for alpha are 0.05 and 0.01.

 '''




print("Part 4 of classification part")

for lr in trained_lrs:
    coefficients = lr.coef_
    print(f'LR coeff : -> {coefficients}')
