#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
from sklearn.metrics import confusion_matrix, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import KBinsDiscretizer

# Import dataset
df1 = pd.read_csv("datasets/jm1.csv")

# Fix invalid values
df1.dropna(inplace=True)
df1.drop_duplicates(inplace=True)

# Print info
df1.info()

# Split up the data set into the features and the labels.
X = df1.drop('defects', axis=1)  # Remove the ___ label.

# Remove outliers and the no-relation values
X.drop('e', axis=1, inplace=True)
X.drop('l', axis=1, inplace=True)

# Discretize dataset
est = KBinsDiscretizer(n_bins=4)
est.fit(X)
X = est.transform(X)

y = df1['defects']

# Split the data set up into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = ComplementNB()
model.fit(X_train.toarray(), y_train)  # Train the model on the training data.
y_pred = model.predict(X_test.toarray())
print(f'Number of mislabeled points out of a total {X_test.shape[0]} points : {(y_test != y_pred).sum()}')

# Calculate precision, recall and f1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')

# Plot the precision recall curve
plot_precision_recall_curve(model, X_test.toarray(), y_test)
