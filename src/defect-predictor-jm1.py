#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


useScaler = True
verbose = 0

df1 = pd.read_csv("datasets/jm1.csv")

# xs = df1[['defects']]
# ys = pd.Series(df1['New cases'])
#
# pd.to_numeric(ys)
#
# # let's visualize it
# plt.xlabel('New recovered')
# plt.ylabel('New cases')
# plt.scatter(xs,ys)

# Fix invalid values
dropped_na_df1 = df1.dropna()
dropped_duplicates_df1 = dropped_na_df1.drop_duplicates()

df1 = dropped_duplicates_df1

df1.info()

# Outliers
plt.boxplot(df1)
plt.show()

# Heatmap
plt.figure(figsize=(12, 10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Split up the data set into the features and the labels.
X = df1.drop('defects', axis=1)  # Remove the ___ label.
y = df1['defects']  # Only take out the ___ label.

# X = input
# y = output (defects)

# Optionally drop features to see how this influences the result.
# X.drop('e', axis=1, inplace=True)
X.drop('l', axis=1, inplace=True)

# Split the data set up into a training and a test set. We can do whatever we want with the training set, but we may only use the test set once, to check the performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data, scaling features. For most regression methods this has no influence, but for some it does.
if useScaler:
    scaler = StandardScaler()  # Create a basic scaler object.
    scaler.fit(X_train)  # Examine the data to find means and standard deviations.
    X_train = scaler.transform(X_train)  # Normalize the data set.
    X_train = pd.DataFrame(X_train,
                           columns=X.columns)  # The scaler only returns the numbers, so we manually turn the numbers into a Pandas dataframe again, with the right column titles.

# Set up a validation set to tune the parameters on.
Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Set up our own evaluation function.
def evaluateResults(title, model, Xt, Xv, yt, yv):
    pred_train = model.predict(Xt)  # Predict values from the training set to determine the fit.
    pred_val = model.predict(Xv)  # Predict values from the validation set to determine the fit.
    RMSE_train = np.sqrt(mean_squared_error(yt, pred_train))
    R2_train = r2_score(yt, pred_train)
    RMSE_val = np.sqrt(mean_squared_error(yv, pred_val))
    R2_val = r2_score(yv, pred_val)
    print("The performance of {} on the training and validation set is:".format(title))
    print("Training set: RMSE = {}, R2 = {}".format(RMSE_train, R2_train))
    print("Validation set: RMSE = {}, R2 = {}\n".format(RMSE_val, R2_val))


# Try linear regression.
model = LinearRegression()  # Set up a regression model.
model.fit(Xt, yt)  # Train the model on the training data.
evaluateResults('linear regression', model, Xt, Xv, yt, yv)

print('fin')