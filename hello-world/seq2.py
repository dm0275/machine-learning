#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

sentences = [
    "I love this product!",
    "This is terrible.",
    "Amazing experience!",
    "I'm not satisfied.",
    "Best purchase ever!"
]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

max_len = max([len(seq) for seq in sequences])

model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
y_train = labels
model.fit(x_train, y_train, epochs=10)

new_sentences = [
    "This is fantastic!",
    "I'm disappointed."
]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
x_new = keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=max_len)
predictions = model.predict(x_new)
for sentence, prediction in zip(new_sentences, predictions):
    sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
    print(f"Sentence: {sentence}, Sentiment: {sentiment}, Confidence: {prediction[0]:.4f}")