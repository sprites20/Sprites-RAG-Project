from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize

sentences = ["Hello!", "Hi!", "How are you?", "What is your name?", "How's the weather today?", "I like pizza.", "I love pizza.", "Python is a great programming language.", "The quick brown fox jumps over the lazy dog.", "To be or not to be, that is the question."]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]

# Normalize embeddings
sentence_embeddings_scaled = sentence_embeddings / 20.0
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Compute cosine similarity matrix
cosine_similarity_matrix = np.dot(sentence_embeddings_scaled, sentence_embeddings_scaled.T)

# Compute Euclidean distance matrix
euclidean_distance_matrix = euclidean_distances(sentence_embeddings_scaled)

# Plot both matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(cosine_similarity_matrix, annot=True, xticklabels=sentences, yticklabels=sentences, cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Cosine Similarity Matrix")

sns.heatmap(euclidean_distance_matrix, annot=True, xticklabels=sentences, yticklabels=sentences, cmap="YlGnBu_r", ax=axes[1], fmt='.2f')
axes[1].set_title("Euclidean Distance Matrix (Inverted)")

plt.tight_layout()
plt.show()

# Normalize cosine similarity matrix
cosine_similarity_matrix_normalized = (cosine_similarity_matrix - np.min(cosine_similarity_matrix)) / (np.max(cosine_similarity_matrix) - np.min(cosine_similarity_matrix))

# Normalize Euclidean distance matrix
euclidean_distance_matrix_normalized = 1 - (euclidean_distance_matrix - np.min(euclidean_distance_matrix)) / (np.max(euclidean_distance_matrix) - np.min(euclidean_distance_matrix))

# Plot both normalized matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(cosine_similarity_matrix_normalized, annot=True, xticklabels=sentences, yticklabels=sentences, cmap="YlGnBu", ax=axes[0], fmt='.2f')
axes[0].set_title("Normalized Cosine Similarity Matrix")

sns.heatmap(euclidean_distance_matrix_normalized, annot=True, xticklabels=sentences, yticklabels=sentences, cmap="YlGnBu", ax=axes[1], fmt='.2f')
axes[1].set_title("Normalized Euclidean Distance Matrix (Inverted)")

plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans

# Normalize embeddings
normalized_embeddings = sentence_embeddings / 20.0  # Divide by 20 as requested

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(normalized_embeddings)

# Plot the clusters
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(x=normalized_embeddings[:, 0], y=normalized_embeddings[:, 1], hue=clusters, palette='viridis', legend='full')

# Add labels for each point
for i, label in enumerate(sentences):
    scatter.text(normalized_embeddings[i, 0], normalized_embeddings[i, 1], label, fontsize=9)

plt.title("K-means Clustering of Sentence Embeddings")
plt.show()

# Compute Euclidean distance
euclidean_distance = torch.norm(sentence_embeddings[0] - sentence_embeddings[1], p=2)

# Print the Euclidean distance
print("Euclidean distance:", euclidean_distance.item())