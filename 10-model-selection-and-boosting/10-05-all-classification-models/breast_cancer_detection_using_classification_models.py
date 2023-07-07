"""
##########################################################
10.05 Breast Cancer Detection using Classification Models
##########################################################
"""

#----------------------------- Importing the libraries ---------------------------------#
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

#------------------------------- Importing the dataset ---------------------------------#
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#------------------------ Label encoding the dependent variable ------------------------#
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#--------------- Splitting the dataset into the Training set and Test set --------------#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#----------------------------------- Feature scaling -----------------------------------#
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#------------------- Defining each classification model as a function ------------------#

# 1. Logistic Regression
def log_reg():
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier

# 2. K-Nearest Neighbors
def k_nn():
    from sklearn.neighbors import KNeighborsClassifier
    # p = 2 for Euclidean (p = 1 for Manhattan)
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    return classifier

# 3. Support Vector Machine
def svm():
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier

# 4. Kernel SVM
def kernel_svm():
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)   
    return classifier

# 5. Naive Bayes
def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

# 6. Decision Tree
def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier

# 7. Random Forest
def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier

# 8. XGBoost
def xgboost():
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier

# 9. CatBoost
def catboost():
    from catboost import CatBoostClassifier
    # Suppress iteration output with 'Silent' logging level
    classifier = CatBoostClassifier(logging_level="Silent")
    classifier.fit(X_train, y_train)
    return classifier

#------------------------------ Making the Confusion Matrix ----------------------------#
def confusion_matrix(classifier):
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}")
    accuracy_score(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc_score}")

#--------------------------- Applying k-Fold Cross Validation-------------------------- #
def k_fold_cv(classifier):
    cv = 10
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = cv)
    print(f"With {cv}-fold cross val:")
    print(f"    Accuracy: {accuracies.mean()*100:.2f}")
    print(f"    Standard Deviation: {accuracies.std()*100:.2f}")
    
#------------- Displaying performance results of each classification model -------------#
models = {
    "Logistic Regression": log_reg(), 
    "K-Nearest Neighbors": k_nn(),
    "Support Vector Machine": svm(), 
    "Kernel SVM": kernel_svm(),
    "Naive Bayes": naive_bayes(),
    "Decision Tree": decision_tree(),
    "Random Forest": random_forest(),
    "XGBoost": xgboost(),
    "CatBoost": catboost()
}

for model_name, model_function in models.items():
    print(f"Model: {model_name}")
    classifier = model_function
    confusion_matrix(classifier)
    k_fold_cv(classifier)
    print("-"*30)
