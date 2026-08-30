import os
import shutil
import tempfile
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma


def get_embedding_function(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    Get the embedding function based on selected provider.
    Supports Google Gemini embeddings and local HuggingFace embeddings.
    """
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API Key is required for Gemini embeddings. "
                "Please provide it in the UI or set GOOGLE_API_KEY in your .env file."
            )
        return GoogleGenerativeAIEmbeddings(
            model=model_name or "models/text-embedding-004",
            google_api_key=api_key,
        )
    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )
    elif provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(
            model=model_name or "nomic-embed-text",
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def build_vector_store(
    chunks: List[Document],
    embedding_function: Embeddings,
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_documents",
):
    """
    Index document chunks into a ChromaDB or In-Memory vector store.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunks list.")
        
    try:
        from langchain_chroma import Chroma
        if persist_directory:
            return Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        else:
            return Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                collection_name=collection_name,
            )
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma
            if persist_directory:
                return Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_function,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )
            else:
                return Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_function,
                    collection_name=collection_name,
                )
        except Exception:
            # High-performance lightweight In-Memory vector store fallback
            from langchain_core.vectorstores import InMemoryVectorStore
            vector_store = InMemoryVectorStore(embedding_function)
            vector_store.add_documents(chunks)
            return vector_store


def load_persisted_vector_store(
    embedding_function: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_documents",
):
    """
    Load an already indexed and persisted ChromaDB vector store from local disk.
    """
    if not os.path.exists(persist_directory):
        return None
        
    try:
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function,
                collection_name=collection_name,
            )
        except Exception:
            return None


def clear_persisted_vector_store(persist_directory: str = "./chroma_db"):
    """
    Delete persisted vector store files from disk when resetting.
    """
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except Exception:
            pass


def get_retriever(
    vector_store,
    search_type: str = "similarity",
    k: int = 4,
):
    """
    Returns a retriever instance configured with search parameters.
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
