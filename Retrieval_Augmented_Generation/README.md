# RAG pipeline with Attention Extraction (Gemma)

This project implements a retrieval-augmented generation (RAG) pipeline for recommending sensors while extracting attention matrices from a large language model (Gemma). The attention outputs are saved for analysis and comparison with other hallucination mitigation methods.

The system is designed to run in Google Colab with GPU support and uses a FAISS HNSW vector store built from technical PDF documents.

## Overview

1. (RAG_database.ipynb)     Load and chunk technical PDF documents                          

2. (RAG_database.ipynb)     Embed text chunks using a sentence transformer           

3. (RAG_database.ipynb)     Store embeddings in a FAISS HNSW index                       

4. (RAG.ipynb)           Retrieve top-k relevant context for each query

5. (RAG.ipynb)           Generate a sensor recommendation using Gemma

6. (RAG.ipynb)           Extract and save attention matrices from the final decoding step

## Data & Directory Structure
Capstone/
├── Data/Documents/C5ISR/      # Source PDFs
├── RAG_Data/Vectorstore_Final/
│   ├── index.faiss
│   └── index.pkl
├── mines.csv
└── RAG_Output/
    └── YYYY-MM-DD_HH-MM-SS/
        ├── sensor_recommendations.csv
        └── attn/*.json

## Environment

Tested in Google Colab with:

PyTorch 2.8.0 + CUDA 12.6

Transformers 4.57.1

NumPy 2.0.2

LangChain 0.3.x

FAISS 1.13.0

Sentence-Transformers 5.1.2

## Usage (Google Colab)

Enable GPU:
Runtime → Change runtime type → GPU

Add Hugging Face token:
Runtime → Secrets → HUGGINGFACE_TOKEN

Mount Google Drive (handled in notebook)

Run RAG_database.ipynb

Run RAG.ipynb

## Notes & Limitations

Attention extraction significantly increases runtime (≈10–100×)

Attention values are model internal proxies, not ground truth

Results depend on model version, decoding settings, and prompt format

### Author: Aquiles Elbaum