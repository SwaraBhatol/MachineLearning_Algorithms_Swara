# -*- coding: utf-8 -*-
"""1_logistic_regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oVtS3QL59uYcd28oV9kjqOP_qCoLJpfI
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1_logistic_regression.py
# Logistic Regression on the Iris dataset

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df= pd.read_csv("/content/Iris.csv")

df.shape

df.head()

df.describe()

df.info()

df.drop("Id",axis=1,inplace=True)
df.head()

df.isnull()

print(df["Species"].value_counts())
sns.countplot(df["Species"])

sns.FacetGrid(df, hue="Species", height=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()

print(df.columns)

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy on Iris dataset:", accuracy)

model = LogisticRegression()
model.fit(X_test, y_test)

y_pred = model.predict(X_train)

accuracy = accuracy_score(y_train, y_pred)
print("Logistic Regression Accuracy on Iris dataset:", accuracy)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Iris Logistic Regression")
plt.show()