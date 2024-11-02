import fitz  # PyMuPDF
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

# Directory to store embeddings and clusters
directory = 'documents'
os.makedirs(directory, exist_ok=True)

# Load model and tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3')
model.eval()

def extract_text_and_images_from_pdf(pdf_file):
    """Extract text and images from a PDF file."""
    text = ""
    images = []
    
    # Open the PDF file
    doc = fitz.open(pdf_file)
    
    # Iterate over each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text
        text += page.get_text()
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)  # Store the raw image bytes

    doc.close()
    return text, images

def chunk_tokens(tokens, max_length=4000, overlap_size=20):
    """Chunk tokens into smaller segments with overlap."""
    chunks = []
    chunk_indices = []

    for i in range(0, len(tokens), max_length - overlap_size):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)
        start_idx = i
        end_idx = min(i + max_length, len(tokens))
        chunk_indices.append((start_idx, end_idx))

    return chunks, chunk_indices

def compute_centroid(chunks):
    """Calculate the centroid of the given chunks."""
    if not chunks:
        return None
    embeddings = np.array([chunk['embedding'] for chunk in chunks])
    return embeddings.mean(axis=0)

def create_clusters(document_data, max_chunks_per_cluster=1000):
    """Create clusters of chunk indices from the document data."""
    clusters = []
    current_cluster = []
    cluster_id = 0  # Initialize cluster ID

    # Generate clusters
    for i, chunk in enumerate(document_data["chunks"]):
        doc_id = 1  # Assuming doc_id is 1 for now; update as needed
        chunk_tuple = (doc_id, i)
        
        if chunk_tuple not in current_cluster:
            current_cluster.append(chunk_tuple)

        if len(current_cluster) >= max_chunks_per_cluster:
            # Store the current cluster with its ID and compute the centroid
            cluster_entry = {
                "id": cluster_id,
                "parent_id": None,
                "chunks": current_cluster,
                "centroid": None,  # Will be calculated later
                "children": []
            }
            clusters.append(cluster_entry)
            cluster_id += 1  # Increment cluster ID
            current_cluster = []  # Reset for new cluster

    # Add the last cluster if not empty
    if current_cluster:
        cluster_entry = {
            "id": cluster_id,
            "parent_id": None,
            "chunks": current_cluster,
            "centroid": None,
            "children": []
        }
        clusters.append(cluster_entry)

    # Calculate centroids for each cluster
    for cluster in clusters:
        cluster["centroid"] = compute_centroid([
            {"embedding": document_data["chunks"][chunk_index]["embedding"]}
            for doc_id, chunk_index in cluster["chunks"]
        ])

    return clusters

def save_clusters(clusters):
    """Save the clusters to a file or database."""
    for cluster in clusters:
        print(f"Cluster ID: {cluster['id']}")
        print(f"Parent ID: {cluster['parent_id']}")
        print(f"Chunks: {cluster['chunks']}")  # Should be a list of tuples (doc_id, chunk_index)
        print(f"Centroid: {cluster['centroid']}")
        print(f"Children: {cluster['children']}\n")

def assign_chunks_to_clusters(chunks, clusters):
    """Assign chunks to the closest cluster based on the centroid."""
    for chunk in chunks:
        closest_cluster = None
        closest_distance = float('inf')
        
        for cluster in clusters:
            # Calculate Euclidean distance from chunk to cluster centroid
            distance = np.linalg.norm(chunk["embedding"] - cluster["centroid"])
            if distance < closest_distance:
                closest_distance = distance
                closest_cluster = cluster

        # Assign chunk to the closest cluster
        if closest_cluster is not None:
            closest_cluster["chunks"].append((1, chunks.index(chunk)))  # Assuming doc_id is 1

# PDF file path
pdf_file = "C:/Users/NakaMura/Documents/Code Optimzer with o1.pdf"
text, images = extract_text_and_images_from_pdf(pdf_file)

# Tokenize the entire document
tokens = tokenizer.tokenize(text)
token_offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']

# Define maximum token length and overlap size
max_length = 4000  # Maximum number of tokens in each chunk
overlap_size = 20  # Overlap size in tokens

# Chunk the tokens
chunks, chunk_indices = chunk_tokens(tokens, max_length, overlap_size)

# Get character indices for chunks
char_indices = [(token_offsets[start_idx][0], token_offsets[end_idx - 1][1]) for start_idx, end_idx in chunk_indices]

# Convert token chunks back to strings for processing
chunk_strings = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

# Store embeddings for the document
document_data = {"chunks": []}

# Process each chunk to generate embeddings
for i, (start_idx, end_idx) in enumerate(char_indices):
    print(f"Chunk {i + 1}: Start char index: {start_idx}, End char index: {end_idx}")
    print(chunk_strings[i])
    
    # Tokenize sentences
    encoded_input = tokenizer(chunk_strings[i], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]  # This should give you (1, embedding_dim)
        sentence_embeddings_scaled = sentence_embeddings.numpy() / 20.0  # Scale the embedding

    # Store chunk data
    document_data["chunks"].append({
        "start_id": start_idx,
        "end_id": end_idx,
        "embedding": sentence_embeddings_scaled,
    })

# Create clusters
clusters = create_clusters(document_data, max_chunks_per_cluster=1000)

# Now, assign chunks to their closest clusters based on the centroid
assign_chunks_to_clusters(document_data["chunks"], clusters)

# Save the clusters to a binary file
save_clusters(clusters)
