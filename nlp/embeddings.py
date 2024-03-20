#!/usr/bin/env python

import tensorflow_hub as hub

# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# A list of sentences to be embedded.
sentences = [
    "This is a sentence.",
    "This is another sentence.",
]

# Generate embeddings for each sentence in the list
sentence_embeddings = embed(sentences)

print(sentence_embeddings)
