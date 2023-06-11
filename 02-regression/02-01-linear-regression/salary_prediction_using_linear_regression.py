"""
#################################################
02.01 - Salary Prediction using Linear Regression
#################################################
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
regressor.predict(X_test)

# Visualizing the training set results
plt.scatter(X_train, y_train, c = "red")
plt.plot(X_train, regressor.predict(X_train), c = "blue")
plt.title("Experience vs. Salary (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing the test set results
plt.scatter(X_test, y_test, c = "red")
plt.plot(X_train, regressor.predict(X_train), c = "blue")
plt.title("Experience vs. Salary (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Predicting the salary for an employee with a given number of years of experience
print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_) # b1 = regression coefficient
print(regressor.intercept_) # b0 = regression constant