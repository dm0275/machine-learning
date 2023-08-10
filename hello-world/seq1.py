#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers, models

# Generate a data set
numbers = list(range(1, 10000))

# Create odd/even labels for each number
labels = [num % 2 for num in numbers]  # 0 for even, 1 for odd

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train = numbers
y_train = labels
model.fit(x_train, y_train, epochs=100)

loss, accuracy = model.evaluate(x_train, y_train)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

predictions = model.predict([2, 15, 30, 50, 99])
for num, prediction in zip([2, 15, 30, 50, 99], predictions):
    print(f"Number: {num}, Odd Probability: {prediction[0]:.4f}")