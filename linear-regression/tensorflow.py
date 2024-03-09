#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression

# Test data
# .reshape(1, -1) is needed bc np.array expects a 2D array
train_input = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
train_output = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0])

# Validate data
validate_input = np.array([17.0, 18.0, 19.0, 20.0]).reshape(-1, 1)
validate_output = np.array([52.0, 55.0, 58.0, 61.0])

# Create and train the linear regression model
model = LinearRegression()
model.fit(train_input, train_output)

# Predict outputs for new input data
new_input_x = np.array([5.0, 6.0]).reshape(-1, 1)  # New input data for prediction
predicted_output_y = model.predict(new_input_x)

print("Predicted outputs for new input data:", predicted_output_y)