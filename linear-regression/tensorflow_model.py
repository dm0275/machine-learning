#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# Test data
train_input = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
train_output = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Validate data
validate_input = np.array([17.0, 18.0, 19.0, 20.0], dtype=float)
validate_output = np.array([52.0, 55.0, 58.0, 61.0], dtype=float)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(train_input, train_output, epochs=500, verbose=0)

# Evaluate the model
loss = model.evaluate(validate_input, validate_output, verbose=0)
print("Final loss:", loss)

# Make predictions
predict_input = np.array([5.0, 6.0], dtype=float)
predictions = model.predict(predict_input, verbose=0)
print("Predictions:", predictions.flatten())
