# linear_regression.py

# exercise 5.2.4
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm
import numpy as np

#gets data from heart_project
import heart_project as data


#get data
#cont attributes are [0,3,4,7,9]
X_og, y_og, attribute_names = data.get_data_matrix()

not_cont_att = [1, 2, 5, 6, 8, 10]
cont_attributes = attribute_names
for index in sorted(not_cont_att, reverse=True):
    cont_attributes = np.delete(cont_attributes, index)
    
y_og = y_og.reshape(-1, 1)
X = np.concatenate((X_og, y_og), axis = 1)

for att in cont_attributes:
    attr = np.where(attribute_names == att)
    attr_ind = attr[0][0]
    y = X_og[:, attr_ind]
    X_cols = list(range(0,attr_ind)) + list(range(attr_ind+1,len(attribute_names)))
    X = X_og[:,X_cols]

    # Fit ordinary least squares regression model
    model = lm.LinearRegression()
    model.fit(X,y)

    # Predict alcohol content
    y_est = model.predict(X)
    residual = y_est-y

    # Display scatter plot
    figure()
    subplot(2,1,1)
    plot(y, y_est, '.')
    x_lab = att + '(true)'
    y_lab = att + '(estimated)'
    xlabel(x_lab); ylabel(y_lab);
    subplot(2,1,2)
    hist(residual,40)
    show()

print('Ran Exercise 5.2.4')
