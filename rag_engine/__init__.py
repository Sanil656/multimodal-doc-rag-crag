"""
RAG Engine package for multi-page document processing, CRAG, market data, and visual comparisons.
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
from .crag import run_crag_pipeline, stream_crag_pipeline, grade_document_relevance, rewrite_query
from .vision_comparator import analyze_and_compare_image_with_book
from .market_data import (
    fetch_live_ticker_data,
    create_interactive_candlestick_chart,
    render_candlestick_image_bytes,
    calculate_market_summary,
)

__all__ = [
    "load_document_from_bytes",
    "load_document_from_path",
    "split_documents_into_chunks",
    "build_vector_store",
    "get_retriever",
    "get_embedding_function",
    "create_rag_chain",
    "format_chat_history",
    "get_llm",
    "run_crag_pipeline",
    "stream_crag_pipeline",
    "grade_document_relevance",
    "rewrite_query",
    "analyze_and_compare_image_with_book",
    "fetch_live_ticker_data",
    "create_interactive_candlestick_chart",
    "render_candlestick_image_bytes",
    "calculate_market_summary",
]
