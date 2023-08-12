#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# Define input (x) and output (y) data
input_x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
output_y = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Build a sequential model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(), loss='mean_squared_error')

# Train the model
model.fit(input_x, output_y, epochs=50)

# Save the model
model.save('x_y_model.h5')  # Unable to load the model w/ a '.keras' extension (after saving) due to this issue:
# https://github.com/keras-team/keras/issues/18278
