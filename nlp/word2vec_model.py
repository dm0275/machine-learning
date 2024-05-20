from gensim.models import Word2Vec
from gensim.test.utils import common_texts

# Train a basic Word2Vec model
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector for a word
vector = model.wv['computer']  # Example: Get the embedding for the word 'computer'

# Find most similar words
similar_words = model.wv.most_similar('computer', topn=10)

print("Vector for 'computer':", vector[:10])  # Show first 10 elements for brevity
print("Similar words to 'computer':", similar_words)
