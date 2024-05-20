#!/usr/bin/env python

import os
import pandas as pd
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Import the data set
cwd = os.getcwd()

# read_csv() reads data from a CSV into a DataFrame
data_frame = pd.read_csv(cwd + "/netflix_titles.csv")

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Generate embeddings for each description
descriptions = data_frame['description'].tolist()
embeddings = embed(descriptions)

# Normalize the embeddings
normalized_embeddings = normalize(embeddings)

# Use K-Means clustering on the embeddings (adjust n_clusters as needed)
kmeans = KMeans(n_clusters=2, random_state=42).fit(normalized_embeddings)

# Add cluster labels to the original dataframe
data_frame['cluster'] = kmeans.labels_

# ---------------

# Step 1: Dimensionality Reduction with PCA
pca = PCA(n_components=50)
pca_result = pca.fit_transform(normalized_embeddings)

# Further reduction with t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result)

# Assuming 'kmeans' is your KMeans model from the previous steps

# Step 2: Visualize the clusters
plt.figure(figsize=(12,8))
plt.scatter(tsne_results[:,0], tsne_results[:,1], c=kmeans.labels_, cmap='viridis', marker='o', alpha=0.5)
plt.colorbar()
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Text Data Clustering Visualization with t-SNE')
plt.show()