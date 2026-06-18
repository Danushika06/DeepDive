"""Retrieval layer: live web search -> page fetch -> chunk -> embed -> FAISS retrieve.

This is the RAG half of the pipeline (course Module 3 Tool use + RAG). Groq has no
embeddings endpoint, so embeddings are local sentence-transformers (``index.py``).
"""
