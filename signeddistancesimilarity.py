"""
RAG

Split into chunks

Embed chunks

Cluster Embeddings

Get Cluster Centroid

Each cluster has a centroid
"""
import spacy
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "This is a sample text that we want to split into chunks. Another sentence here."

# Split text into chunks
chunks = [chunk.text for chunk in nlp(text).sents]

# Embed chunks (using a dummy Word2Vec model)
model = Word2Vec(chunks, vector_size=100, window=5, min_count=1, sg=0)
embeddings = np.array([model.wv[chunk] for chunk in chunks])

# Cluster embeddings
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
cluster_indices = kmeans.fit_predict(embeddings)

# Get cluster centroids
cluster_centroids = []
for i in range(num_clusters):
    cluster_embeddings = embeddings[cluster_indices == i]
    centroid = np.mean(cluster_embeddings, axis=0)
    cluster_centroids.append(centroid)

# Euclidean distance between centroids
centroid_distances = euclidean_distances(cluster_centroids, cluster_centroids)
print("Euclidean distance between centroids:")
print(centroid_distances)

# Max Euclidean distance from each centroid in its cluster
max_distances = []
for i, centroid in enumerate(cluster_centroids):
    cluster_embeddings = embeddings[cluster_indices == i]
    distances = euclidean_distances(cluster_embeddings, [centroid])
    max_distance = distances.max()
    max_distances.append(max_distance)
print("Max Euclidean distance from each centroid in its cluster:")
print(max_distances)

# Signed distance from the hypersphere for each point
signed_distances = []
for i, centroid in enumerate(cluster_centroids):
    cluster_embeddings = embeddings[cluster_indices == i]
    distances = euclidean_distances(cluster_embeddings, [centroid])
    radius = max_distances[i]  # Using the max distance as the radius
    signed_distance = distances - radius
    signed_distances.extend(signed_distance.flatten())
print("Signed distance from the hypersphere for each point:")
print(signed_distances)