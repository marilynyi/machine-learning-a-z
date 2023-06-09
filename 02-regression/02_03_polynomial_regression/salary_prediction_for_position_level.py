"""
######################################################################
02.03 Salary Prediction for Position Level using Polynomial Regression
######################################################################
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values # Position index is excluded
y = dataset.iloc[:, -1].values

# Training the Linear Regression model 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y) # using the whole dataset (no training and test datasets)

# Visualizing the Linear Regression results
y_pred = lin_reg.predict(X)
plt.scatter(X, y, c = "red")
plt.plot(X, lin_reg.predict(X), c = "blue")
plt.title("Position Level vs. Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
X_new = 7 # arbitrarily choosing position level 7 to predict their salary)
y_new = lin_reg.predict([[X_new]])
print(f"The predicted salary for level {X_new} using Linear Regression is ${int(y_new):,}")

level = dataset.loc[dataset["Level"] == X_new, :]
print(f"The data row for Level {X_new}: \n{level}")

print(f"Predicted salary / Actual salary using Linear Regression = {round(370818/200000,2)}")

# Training the Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures

nth_degree = 4 # arbitrary maximum coefficient power -> y = b0 + b1x1 + b2x1^2 + ...
poly_reg = PolynomialFeatures(degree = nth_degree) 
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Polynomial Regression results
plt.scatter(X, y, c = "red")
plt.plot(X, lin_reg_2.predict(X_poly), c = "blue")
plt.title("Position Level vs. Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Position Level vs. Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Polynomial Regression
X_new = 7 # arbitrarily choosing position level 7 to predict their salary)
y_new = lin_reg_2.predict(poly_reg.fit_transform([[X_new]]))
print(f"The predicted salary for level {X_new} using Polynomial Regression is ${int(y_new):,}")

level = dataset.loc[dataset["Level"] == X_new, :]
print(f"The data row for Level {X_new}: \n{level}")

print(f"Predicted salary / Actual salary using Polynomial Regression = {round(184003/200000,2)}")
