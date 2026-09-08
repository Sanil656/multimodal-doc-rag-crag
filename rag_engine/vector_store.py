"""
High-performance Vector Store and Caching layer with ChromaDB and In-Memory acceleration.
"""

import os
import shutil
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Global in-memory embedding cache
_EMBEDDING_CACHE = {}


def get_embedding_function(
    provider: str = "huggingface",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    Returns cached embedding instance to prevent redundant re-initialization latency.
    """
    cache_key = f"{provider}_{model_name}_{bool(api_key)}"
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API Key is required for Gemini embeddings.")
        embed_fn = GoogleGenerativeAIEmbeddings(model=model_name or "models/text-embedding-004", google_api_key=key)

    elif provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings
        embed_fn = OllamaEmbeddings(model=model_name or "nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    else:  # Local HuggingFace (Ultra-fast CPU embeddings)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embed_fn = HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

    _EMBEDDING_CACHE[cache_key] = embed_fn
    return embed_fn


def build_vector_store(
    chunks: List[Document],
    embedding_function: Embeddings,
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_documents",
):
    """
    Fast vector indexing with C++ HNSW graph in ChromaDB or In-Memory acceleration.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunks list.")

    try:
        from langchain_chroma import Chroma
        return Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma
            return Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        except Exception:
            from langchain_core.vectorstores import InMemoryVectorStore
            store = InMemoryVectorStore(embedding_function)
            store.add_documents(chunks)
            return store


def load_persisted_vector_store(
    embedding_function: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_documents",
):
    """
    Load persisted ChromaDB from disk if exists.
    """
    if not os.path.exists(persist_directory):
        return None
    try:
        from langchain_chroma import Chroma
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_function, collection_name=collection_name)
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma
            return Chroma(persist_directory=persist_directory, embedding_function=embedding_function, collection_name=collection_name)
        except Exception:
            return None


def clear_persisted_vector_store(persist_directory: str = "./chroma_db"):
    """Clean vector store cache from disk."""
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except Exception:
            pass


def get_retriever(vector_store: Any, k: int = 5):
    """
    Returns an optimized retriever with fast top-k similarity search.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
