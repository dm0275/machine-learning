#!/usr/bin/env python

import numpy as np

# Define input (x) and output (y) data
input_x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
output_y = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Initialize model parameters (weights and bias)
weight = 0.0
bias = 0.0

# Set hyperparameters
learning_rate = 0.01
epochs = 50

# Training loop
for epoch in range(epochs):
    # Forward pass: Compute predicted y
    predicted_y = weight * input_x + bias

    # Compute and print Mean Squared Error (MSE)
    mse = np.mean((predicted_y - output_y) ** 2)
    print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse}")

    # Backpropagation: Compute gradients
    gradient_weight = np.mean(2 * (predicted_y - output_y) * input_x)
    gradient_bias = np.mean(2 * (predicted_y - output_y))

    # Update model parameters using gradients
    weight -= learning_rate * gradient_weight
    bias -= learning_rate * gradient_bias

# Print the final trained parameters
print(f"Trained weight: {weight}")
print(f"Trained bias: {bias}")