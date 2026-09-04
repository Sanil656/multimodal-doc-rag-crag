"""
RAG Engine package for multi-page document processing, LangGraph CRAG, Token Optimization, and vector storage.
"""

from .document_loader import load_document_from_bytes, load_document_from_path
from .text_splitter import split_documents_into_chunks
from .vector_store import (
    build_vector_store,
    get_retriever,
    get_embedding_function,
    load_persisted_vector_store,
    clear_persisted_vector_store,
)
from .chain import create_rag_chain, format_chat_history, get_llm
from .crag_graph import create_crag_graph, stream_langgraph_crag_pipeline, GraphState
from .token_optimizer import (
    count_tokens,
    calculate_cost,
    compress_and_prune_documents,
    evaluate_groundedness,
)

__all__ = [
    "load_document_from_bytes",
    "load_document_from_path",
    "split_documents_into_chunks",
    "build_vector_store",
    "get_retriever",
    "get_embedding_function",
    "load_persisted_vector_store",
    "clear_persisted_vector_store",
    "create_rag_chain",
    "format_chat_history",
    "get_llm",
    "create_crag_graph",
    "stream_langgraph_crag_pipeline",
    "GraphState",
    "count_tokens",
    "calculate_cost",
    "compress_and_prune_documents",
    "evaluate_groundedness",
]
