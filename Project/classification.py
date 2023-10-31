#classification file

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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


X, y, attribute_names = pplib.get_data_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

#BASELINE MODEL: The baseline will be a model which compute the largest class on the training data, and predict everything in the test-data as belonging to that class




#METHOD 2:
#method 2 option 1 :  Dec Tree
clf = DecisionTreeClassifier()  #we can add these to improve accuracy criterion="entropy", max_depth=3
# Train DecTree
clf = clf.fit(X_train,y_train)
#Predict on test dataset
y_pred = clf.predict(X_test)
print("DecTree Accuracy:", accuracy_score(y_test, y_pred)) #Accuracy: 0.7554347826086957

#method 2 option 2 : Knn
for k in range(1, 11):
    clf = KNeighborsClassifier(k)  #iterate with different Ks to improve accuracy
    clf = clf.fit(X_train, y_train) #train
    y_pred = clf.predict(X_test) #test
    print("knn Accuracy:", accuracy_score(y_test, y_pred), "K:", k)

#method 2 option 3: ANN
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,                               hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("NN Accuracy:", accuracy_score(y_test, y_pred))

#method 2 option 4: NB
clf = GaussianNB()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("NB Accuracy:", accuracy_score(y_test, y_pred))

#logistic regression:

#normalization
#X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
#X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

clf = LogisticRegression(max_iter = 5000)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("LR Accuracy:", accuracy_score(y_test, y_pred))


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
