"""
########################################################
07.01 Sentiment Analysis of Restaurant Reviews using NLP
########################################################
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings

nltk.download("stopwords")
warnings.filterwarnings("ignore")

# Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Cleaning the texts
remove_stopwords = ["not", "don't", 'didn', "didn't", 'wasn', "wasn't", 'weren', "weren't", 'wouldn', "wouldn't"]

corpus = []
for i in range(0, len(dataset)):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower().split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    for sw in remove_stopwords:
        all_stopwords.remove(sw)
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
