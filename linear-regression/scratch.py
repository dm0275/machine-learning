#!/usr/bin/env python

import numpy as np


def simple_linear_regression_fit(input_x, output_y, learning_rate, epochs):
    # Initialize parameters (weights and bias)
    np.random.seed(0)

    # The weight and bias parameters define the relationship between the input features and the output in a linear
    # regression model.
    # The weight represents the effect of each input feature on the output
    # weight = np.random.randn()
    weight = 1
    # The bias represents the baseline value of the output when all input features are zero
    # bias = np.random.randn()
    bias = 1

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
        weight = weight - (learning_rate * d_weight)
        bias = bias - (learning_rate * d_bias)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    print("Training finished!")

    return weight, bias


# Test data - f(x) = 3x +1
train_input = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
train_output = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Validate data
validate_input = np.array([17.0, 18.0, 19.0, 20.0], dtype=float)
validate_output = np.array([52.0, 55.0, 58.0, 61.0], dtype=float)

epochs = 500
learning_rate = 0.01

weight, bias = simple_linear_regression_fit(train_input, train_output, learning_rate, epochs)

print("Final weight:", weight)
print("Final bias:", bias)

# Predict output for validation data using the trained model
validate_predicted_output = weight * validate_input + bias

# Calculate validation loss
validation_loss = np.mean((validate_predicted_output - validate_output) ** 2)

print("Validation Loss:", validation_loss)

# Data for prediction
predict_input = np.array([5.0, 6.0], dtype=float).reshape(-1, 1)

# Predict output for the new input data using the trained model
predicted_output_y = weight * predict_input + bias

print("Predicted outputs for new input data:", predicted_output_y)
