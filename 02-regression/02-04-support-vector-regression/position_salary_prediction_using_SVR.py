"""
##########################################################################
02.04 Salary Prediction for Position Level using Support Vector Regression
##########################################################################
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values # Position index is excluded
y = dataset.iloc[:, -1].values

# Reshaping y to same array form as X to prep for feature scaling
y = y.reshape(len(y), 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X, y)

# Predicting a new result
X_level = 7
y_level = sc_y.inverse_transform(regressor.predict(sc_X.transform([[X_level]])).reshape(-1, 1))
print(f"The expected compensation for a Level 7 position is ${int(y_level):,}")

# Visualizing the SVR results
y_new = sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color = "blue")
plt.title("Position Level vs. Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color = "blue")
plt.title("Position Level vs. Salary (high res, smooth)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()