import os
import sys

import numpy
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras import layers

# Import the data set
df = pd.read_csv("data.csv")

# Cleanup
# Drop null items
df = df.dropna()

# Encode str features
df = df.apply(LabelEncoder().fit_transform)

X = df.drop(columns=['price'])
y = df['price']


# Calculate mutual info scores
def calculate_mutual_info_score(X, y):
    X = pd.DataFrame(X)

    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    discrete_features = X.dtypes == int
    return mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)


# Select top features based on the MI score
selected_top_columns = SelectPercentile(calculate_mutual_info_score, percentile=20)
selected_top_columns.fit(X, y)
X = X[X.columns[selected_top_columns.get_support()]]

# Split features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

predict_score = model.score(X_train, y_train)

print(predict_score)
