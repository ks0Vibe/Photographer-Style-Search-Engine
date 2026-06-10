# Photographer Style Search Engine

A multimodal photographer-oriented search engine that retrieves images by visual style, semantic content and photographic characteristics.

## Features

- Image-to-image retrieval
- Text-to-image retrieval
- CLIP embeddings
- Photography visual descriptors
- FAISS indexing
- Style-aware reranking

## Dataset

Unsplash Lite Dataset

## Architecture

Image
→ Feature Extraction
→ Vector Storage
→ FAISS Index
→ Search Engine
→ Style Reranker

## Project Structure

app/
scripts/
data/
experiments/

## Setup

```bash
pip install -r requirements.txt