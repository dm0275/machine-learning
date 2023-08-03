#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

# Load the mnist data set. This data set which consists of 60,000+ grayscale images of handwritten digits for training
# and 10,000 images for testing. TensorFlow provides easy access to this dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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
