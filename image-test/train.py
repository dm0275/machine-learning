#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers, models

# Load the mnist data set.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Rescale the intensity level(brightness) of each image so that they fall with in the range of 0 or 1.
# We're normalizing the image and making it easier for the model to learn and make predictions
# 0 is a completely black pixel and 1 is a completely white pixel
x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

predictions = model.predict(x_test)
print('Prediction: ', predictions)
