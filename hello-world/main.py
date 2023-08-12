#!/usr/bin/env python

from tensorflow import keras

# Load the trained model
loaded_model = keras.models.load_model('x_y_model.h5')

print(loaded_model.predict([4.0]))
