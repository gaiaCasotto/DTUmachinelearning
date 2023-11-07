#classification file

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
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
    
def inner_loop(models, X_train, y_train, CV):
    best_models = {}
    best_accuracies = {}
    for (modelname, model_array) in models.items():
        best_acc = 0;
        for model in model_array:
            print(f'evaluating model : {model}')
            results = cross_val_score(model, X_train, y_train, cv=CV, scoring='accuracy')
            for fold, accuracy in enumerate(results, start=1):
                print(f'Fold {fold} - Accuracy: {accuracy:.2f}')
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_models[modelname] = model
                    best_accuracies[modelname] = best_acc
        #Calculate and print the mean and standard deviation of the accuracy
        mean_accuracy = results.mean()
        std_accuracy = results.std()
        print(f'Mean Accuracy: {mean_accuracy:.2f} +/- {std_accuracy:.2f}')

    print(f'best models -> {best_models.items()}')
    print(f'model accuracies -> {best_accuracies.items()}')
    return best_models  #and possibly best parameters

def outer_loop(best_models, X, y, CV):
    best_model = None
    best_acc = 0;
    for (modelname, model) in best_models.items():
        results = cross_val_score(model, X, y, cv=CV, scoring='accuracy')
        for fold, accuracy in enumerate(results, start=1):
            print(f'Fold {fold} - Accuracy: {accuracy:.2f}')
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = model
                
    print(f'best model is {best_model} with accuracy {best_acc}')
    return best_model, best_acc

 
 
X, y, attribute_names = pplib.get_data_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)

hidden_layer_sizes =  [(50,), (100,), (50, 50), (50, 100), (100, 50), (150), (75)]

neuralNetworks = []
for sizes in hidden_layer_sizes:
    neuralNetworks.append(MLPClassifier(max_iter = 500, hidden_layer_sizes = sizes))
lambdas = [0.001, 0.01, 0.1, 1, 10]
log_regressors = []
for l in lambdas:
    log_regressors.append(LogisticRegression( C = 1/l, max_iter = 5000))
  
models = {}
models['neuralNetworks'] = neuralNetworks
models['LR'] = log_regressors
#models['baseline'] = baseline


K = 10
CV = KFold(n_splits=K, shuffle=True, random_state= 42)

best_models          = inner_loop(models, X_train, y_train, CV)
best_model, best_acc = outer_loop(best_models, X, y, CV)

baseline_y_pred = baseline_model(X_test.shape[0], y_train)
print("Baseline Accuracy:", accuracy_score(y_test, baseline_y_pred))

'''
#METHOD 2: ANN
#we can change the layer size
#or alpha value
#or the solver for weight optimization
clf = MLPClassifier(hidden_layer_sizes = (50))
#clf = MLPClassifier(solver='adam', alpha=1e-5,                               hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("NN Accuracy:", accuracy_score(y_test, y_pred))

#LOGISTIC REGRESSION:

lambda_values = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for a_lambda in range(1, 10):
    lambda_values.append(a_lambda)
# Create a parameter grid for GridSearchCV
param_grid = {'C': [1 / a_lambda for a_lambda in lambda_values]}
# Initialize Logistic Regression
clf = LogisticRegression(max_iter=5000)
# Perform grid search with cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5)  # 5-fold cross-validation
grid_search.fit(X_train, y_train)
# Get the best lambda value from the grid search
best_lambda = 1 / grid_search.best_params_['C']

# Train Logistic Regression with the best lambda
best_clf = LogisticRegression(C=grid_search.best_params_['C'], max_iter=5000)
best_clf = best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)
print(f"Best Lambda: {best_lambda}")
print(f"LR Accuracy with Best Lambda:", accuracy_score(y_test, y_pred))
'''
'''
def cv_report(models, X, y):
    results = []
    for name in models.keys():
        model = models[name]
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Accuracy: %.3f (+/- %.3f) [%s]" %(scores.mean(), scores.std(), name))


#print(X_test.shape)
#print(X_train.shape)



models = {
    'CART': DecisionTreeClassifier(), #this could be method 2
    'SVC' : SVC(probability=True),
    'GNB' : GaussianNB(),
    'LDA' : LinearDiscriminantAnalysis(),
    'KNN' : KNeighborsClassifier()   #this could be method 2
    #also NN could be method 2
}


cv_report(models, X_train, y_train)
'''

'''
#method 2 option 1 :  Dec Tree
clf = DecisionTreeClassifier()  #we can add these to improve accuracy criterion="entropy", max_depth=3
# Train DecTree
clf = clf.fit(X_train,y_train)
#Predict on test dataset
y_pred = clf.predict(X_test)
print("DecTree Accuracy:", accuracy_score(y_test, y_pred)) #Accuracy: 0.7554347826086957
'''
'''
#will not use knn
#method 2 option 2 : Knn
for k in range(1, 11):
    clf = KNeighborsClassifier(k)  #iterate with different Ks to improve accuracy
    clf = clf.fit(X_train, y_train) #train
    y_pred = clf.predict(X_test) #test
    print("knn Accuracy:", accuracy_score(y_test, y_pred), "K:", k)
'''
'''
#method 2 option 4: NB
clf = GaussianNB()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("NB Accuracy:", accuracy_score(y_test, y_pred))
'''
