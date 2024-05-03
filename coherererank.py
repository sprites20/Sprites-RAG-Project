# rerank

from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.cohere import Cohere
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

import os

# Define the URL of the webpage to scrape
url = 'https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
}

# Scrape the webpage and extract the text
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Define chunk size and overlap
chunk_size = 10  # You can adjust this value
overlap = 3     # Number of sentences to overlap

# Split sentences into chunks with overlap
chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences) - overlap, chunk_size - overlap)]

# Save each chunk to a separate text file
os.makedirs("chunk_texts", exist_ok=True)
for i, chunk_data in enumerate(chunks):
    chunk_text = " ".join(chunk_data)
    with open(f"chunk_texts/chunk_{i+1}.txt", "w") as file:
        file.write(chunk_text)

# Set up the Cohere API key
API_KEY = #API_KEY

# Create the embedding model
embed_model = CohereEmbedding(
    cohere_api_key=API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

# Create the service context with the Cohere model for generation and embedding model
service_context = ServiceContext.from_defaults(
    llm=Cohere(api_key=API_KEY, model="command"),
    embed_model=embed_model
)

# Load the data from the saved chunk files
data = SimpleDirectoryReader(input_dir="chunk_texts").load_data()

# Create the index
index = VectorStoreIndex.from_documents(data, service_context=service_context)

# Create the Cohere reranker
cohere_rerank = CohereRerank(api_key=API_KEY)

# Create the query engine
query_engine = index.as_query_engine(node_postprocessors=[cohere_rerank])

# Generate the response
response = query_engine.query("How to print nth fibonacci number.")

print(response)