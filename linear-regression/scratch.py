#!/usr/bin/env python

import numpy as np


def simple_linear_regression_fit(input_x, output_y, learning_rate, epochs):
    # Initialize parameters (weights and bias)
    np.random.seed(0)  # for reproducibility
    weight = np.random.randn()
    bias = np.random.randn()

    print(weight)
    print(bias)

    # Number of samples
    n = len(input_x)

    # Training the model
    for epoch in range(epochs):
        # Forward pass: compute predicted y
        predicted_output_y = weight * input_x + bias

        # Compute loss
        loss = np.mean((predicted_output_y - output_y) ** 2)

        # Backpropagation: compute gradients
        d_weight = (2 / n) * np.sum((predicted_output_y - output_y) * input_x)
        d_bias = (2 / n) * np.sum(predicted_output_y - output_y)

        # Update parameters
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    print("Training finished!")

    return weight, bias


# Test data
train_input = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
train_output = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Validate data
validate_input = np.array([17.0, 18.0, 19.0, 20.0], dtype=float)
validate_output = np.array([52.0, 55.0, 58.0, 61.0], dtype=float)

epochs = 50
learning_rate = 0.01

weight, bias = simple_linear_regression_fit(train_input, train_output, learning_rate, epochs)

print("Final weight:", weight)
print("Final bias:", bias)

# Predict output for validation data using the trained model
validate_predicted_output = weight * validate_input + bias

# Calculate validation loss
validation_loss = np.mean((validate_predicted_output - validate_output) ** 2)

print("Validation Loss:", validation_loss)

# New input data for prediction
new_input_x = np.array([5.0, 6.0])

# Predict output for new input data using the trained model
predicted_output_y = weight * new_input_x + bias

print("Predicted outputs for new input data:", predicted_output_y)
