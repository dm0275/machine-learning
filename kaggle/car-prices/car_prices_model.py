import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model

data_frame = pd.read_csv("car_prices.csv")

# Drop rows with NA columns
data_frame = data_frame.dropna()

# Drop categorical features for this simple model
categorical_features = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color']
data_frame = data_frame.drop(categorical_features, axis=1)

# Drop misc features
misc_features = ['interior', 'seller', 'vin', 'saledate']
data_frame = data_frame.drop(misc_features, axis=1)

inputs = data_frame.drop('sellingprice', axis=1)
output = data_frame.sellingprice

# Split the dataframe into a train and test sets, where we use 20% of the data for testing
X_train,X_test,y_train,y_test = train_test_split(inputs, output, test_size=0.2, random_state=42)

X_train = X_train["odometer"]
X_test = X_test["odometer"]

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Final loss:", loss)