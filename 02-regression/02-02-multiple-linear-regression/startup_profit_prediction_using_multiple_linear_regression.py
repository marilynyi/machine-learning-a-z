"""
###################################################################
02.02 Profit Prediction of Startup using Multiple Linear Regression
###################################################################
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X[:5])

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# columns to be transformed are noted in 3rd index (4th column) of transformers; passthrough to keep the other columns unchanged
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(X[:5])

# Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test) # Instantiate predicted vector of the test set
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis = 1))

# Predicting the profit of a startup with given attributes
# R&D Spend = 160,000
# Administration = 130,000
# Marketing Spend = 300,000
# State = California [1, 0, 0]
print(regressor.predict([[1, 0, 0, 160000,130000,300000]])) 

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_) # b1, b2, ... = regression coefficients
print(regressor.intercept_) # b0 = regression constant