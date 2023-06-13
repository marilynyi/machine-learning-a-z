"""
#########################################################################
02.06 Salary Prediction for Position Level using Random Forest Regression
#########################################################################
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values 
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0) # 10 trees; fixed seed
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[7]])
print(f"The predicted salary for position level 7 is ${int(regressor.predict([[7]])):,}")

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Position Level vs. Salary (Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.ticklabel_format(axis="y", style="plain")
plt.show()