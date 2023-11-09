# Hello World - From Scratch

## Requirements
* `numpy`

## Steps
1. Define the input (`x`) and output (`y`) data
    * ```python
      input_x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
      output_y = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
      ```
    * `input_x`: An array containing the input data points.
    * `input_y`: An array containing the corresponding output data points.
2. Initialize the initial weight and bias for the model
    * ```python
      weight = 0.0
      bias = 0.0
      ```
    * We're initializing both the `weight` and the `bias` with a value of `0.0` because our model is very simple. Complex models would require a different initialization technique such as starting with a random number.
3. Configure the hyperparameters
    * ```python
      learning_rate = 0.01
      epochs = 50
      ```
    * Hyperparameters are external configuration settings for an ML model which are not learnt from the data but pre-determined by the person conducting the training
    * `learning_rate`: The step size used during optimization to update model parameters. 
      * It controls the size of parameter updates during training.
    * `epochs`: The number of times the entire training dataset is passed forward and backward through the neural network during training.
4. Start the training and loop through the training steps based on the number of `epochs` defined above
5. Compute the predicted Y using a forward pass
   * ```python
     predicted_y = weight * input_x + bias
     ```
   * `weight`: This represents the weight parameter of the linear regression model. 
     * The `weight` is a scalar value that determines the strength of the linear relationship between the input `input_x` and the output `predicted_y`.
     * A scalar value is just a single number (ex. 5 or -2) It's a number without any direction, just a size.
     * The `weight` says how much the input numbers influence the predictions. A big scalar means they influence a lot, a small scalar means they influence just a little.
   * input_x: This is the input data, a NumPy array containing a set of input values. Each element of input_x corresponds to one data point.
6. d