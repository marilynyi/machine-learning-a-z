"""
#############################
01 - Data Preprocessing Tools
#############################
"""

# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# IMPORTING THE DATASET
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values # ind variable: all rows, all columns except last
y = dataset.iloc[:, -1].values # dep variable: all rows, last column


# TAKING CARE OF MISSING DATA

# Import SimpleImputer module to take care of missing data
from sklearn.impute import SimpleImputer # class.module 

# Replace empty missing values with the mean in the column
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") 

# Apply transformation of value replacements in only Age and Salary columns
# in practice, include all columns with numerical data
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# ENCODING CATEGORICAL DATA

# Encoding the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# columns to be transformed are noted in 3rd arg of transformers; passthrough to keep the other columns unchanged
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # no args because one vector
y = le.fit_transform(y)


# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) # transform Age and Salary columns only in X_train
X_test[:, 3:] = sc.transform(X_test[:, 3:]) # apply transformation to X_test