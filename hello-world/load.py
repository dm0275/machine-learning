#!/usr/bin/env python

import tensorflow as tf

# Load the trained model
loaded_model = tf.keras.models.load_model('x_y_model.h5')

print(loaded_model.predict([4.0]))
