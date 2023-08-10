# Hello World

## Requirements
* `tensorflow`
* `matplotlib`

## Steps
1. Define the input (`x`) and output (`y`) data
   * ```python
     input_x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
     output_y = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
     ```
   * `input_x`: An array containing the input data points.
   * `input_y`: An array containing the corresponding output data points.
2. Build a sequential model
   * ```python
     model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
     ```
   * `model`: Is a sequential model is created using Keras.
   * A sequential model is a machine learning model that is used to learn from data in a sequential manner
   * A model can be thought of like a mathematical function that represents a relationship between input data and the corresponding output or target
   * `keras.layers.Dense(units=1, input_shape=[1])`: Adds a dense (fully connected) layer to the model. `units=1` indicates that this layer has one neuron and `input_shape=[1]` specifies the shape of the input data.
     * This essentially adds a layer with one neuron that expects a one-dimensional input
     * `units=1`: This parameter specifies the number of neurons or units in the layer. In other words, it determines the dimensionality of the output space of this layer. For example, if you set `units=1`, the layer will have one neuron and produce a single output value.
     * `input_shape=[1]`: This parameter specifies the shape of the input data that will be fed into the layer. The input data shape should match the shape of the data you intend to pass into the network. In this case, the [1] indicates that the layer expects a one-dimensional input with one feature.
3. Compile the model
   * ```python
     model.compile(optimizer='sgd', loss='mean_squared_error')
     ```
   * When we say "compiling the model," this is not referring to the compiling code process in programming. Instead, we're configuring the training settings and preparing the model for the training process.
   * `optimizer='sgd'`: This parameter specifies the optimization algorithm to be used during training. `sgd` stands for Stochastic Gradient Descent, which is a widely used optimization algorithm for training neural networks. Optimization algorithms update the model's parameters (weights) during training to minimize the loss function.
   * `loss='mean_squared_error`: This parameter specifies the loss function that the model will use to measure how well it's performing during training. `mean_squared_error` is a common loss function for regression problems. It computes the average squared difference between the predicted values and the actual target values. The goal of training is to minimize this loss, effectively making the model's predictions as close as possible to the actual targets.
4. TODO

## Notes:
* Layer - A layer is a fundamental building block that processes and transforms data as it flows through the network. Think of a neural network as a stack of layers, each responsible for extracting and learning specific features from the input data.
  * ![](https://image.slidesharecdn.com/10-things-every-php-developer-should-know-about-machine-learning-170409043418/95/10-things-every-php-developer-should-know-about-machine-learning-35-638.jpg?cb=1491712734) 
  * Each layer in a neural network performs a set of operations on the input data and produces an output that becomes the input for the next layer. The layers are connected sequentially, forming the architecture of the network. There are various types of layers, each designed for different purposes:
    1. Input Layer: The first layer of the network, where data is fed into the network. It does not perform any computation but simply passes the data to the next layer.
    2. Hidden Layers: Layers between the input and output layers are called hidden layers. They perform computations and learn features from the data. Different types of layers, such as dense (fully connected) layers, convolutional layers, and recurrent layers, can be used as hidden layers.
    3. Output Layer: The final layer of the network that produces the model's predictions or output. The structure of this layer depends on the problem you're solving, such as classification, regression, etc.