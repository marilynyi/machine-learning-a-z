"""
############################################
10.04 Breast Cancer Detection using CatBoost
############################################
"""

# Importing the libraries
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Label encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training CatBoost on the Training set
from catboost import CatBoostClassifier
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:\n{cm}")
accuracy_score(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc_score}")

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(f"Accuracy: {accuracies.mean()*100:.2f}")
print(f"Standard Deviation: {accuracies.std()*100:.2f}")
