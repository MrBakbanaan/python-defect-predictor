#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# Define settings.
useScaler = True
verbose = 0

df1 = pd.read_csv("corona.csv")
df1.info()
df1.head()

xs = df1[['New recovered']]
ys = pd.Series(df1['New cases'])

pd.to_numeric(ys)

# let's visualize it
plt.xlabel('New recovered')
plt.ylabel('New cases')
plt.scatter(xs,ys)

# Split up the data set into the features and the labels.
X = df1.drop('New cases', axis=1) # Remove the ___ label.
y = df1['New cases'] # Only take out the ___ label.

# Optionally drop features to see how this influences the result.
X = X.drop('Date', axis=1)
#X = X.drop('Confirmed', axis=1)
#X = X.drop('Recovered', axis=1)
#X = X.drop('Active', axis=1)
#X = X.drop('New cases', axis=1)
#X = X.drop('New deaths', axis=1)
#X = X.drop('New recovered', axis=1)
#X = X.drop('Deaths / 100 Cases', axis=1)
#X = X.drop('Recovered / 100 Cases', axis=1)
#X = X.drop('Deaths / 100 Recovered', axis=1)
#X = X.drop('No. of countries', axis=1)



# Split the data set up into a training and a test set. We can do whatever we want with the training set, but we may only use the test set once, to check the performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data, scaling features. For most regression methods this has no influence, but for some it does.
if useScaler:
    scaler = StandardScaler() # Create a basic scaler object.
    scaler.fit(X_train) # Examine the data to find means and standard deviations.
    X_train = scaler.transform(X_train) # Normalize the data set.
    X_train = pd.DataFrame(X_train, columns=X.columns) # The scaler only returns the numbers, so we manually turn the numbers into a Pandas dataframe again, with the right column titles.

# Set up a validation set to tune the parameters on.
Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Set up our own evaluation function.
def evaluateResults(title, model, Xt, Xv, yt, yv):
    pred_train = model.predict(Xt) # Predict values from the training set to determine the fit.
    pred_val = model.predict(Xv) # Predict values from the validation set to determine the fit.
    RMSE_train = np.sqrt(mean_squared_error(yt, pred_train))
    R2_train = r2_score(yt, pred_train)
    RMSE_val = np.sqrt(mean_squared_error(yv, pred_val))
    R2_val = r2_score(yv, pred_val)
    print("The performance of {} on the training and validation set is:".format(title))
    print("Training set: RMSE = {}, R2 = {}".format(RMSE_train, R2_train))
    print("Validation set: RMSE = {}, R2 = {}\n".format(RMSE_val, R2_val))

# Try linear regression.
model = LinearRegression() # Set up a regression model.
model.fit(Xt, yt) # Train the model on the training data.
evaluateResults('linear regression', model, Xt, Xv, yt, yv)

