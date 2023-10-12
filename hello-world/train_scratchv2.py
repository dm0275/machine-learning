import numpy as np

# Define input (x) and output (y) data
input_x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
output_y = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


def gradient_descent_optimizer(input_x, output_y, learning_rate=0.01, epochs=50):
    # Initialize model parameters (weights and bias)
    weight = 0.0
    bias = 0.0

    for epoch in range(epochs):
        # Forward pass: Compute predicted y
        predicted_y = weight * input_x + bias

        # Compute gradients
        gradient_weight = np.mean(2 * (predicted_y - output_y) * input_x)
        gradient_bias = np.mean(2 * (predicted_y - output_y))

        # Update model parameters using gradients and learning rate
        weight -= learning_rate * gradient_weight
        bias -= learning_rate * gradient_bias

    return weight, bias


# Training using gradient descent optimizer
final_weight, final_bias = gradient_descent_optimizer(input_x, output_y)

# Print the final trained parameters
print(f"Trained weight: {final_weight}")
print(f"Trained bias: {final_bias}")