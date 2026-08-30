from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks while preserving page and source metadata.
    
    Args:
        documents: List of Document objects with metadata (page, source, etc.)
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked Document objects with chunk_id metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    
    # Enrich metadata with chunk identifiers
    for idx, doc in enumerate(chunked_docs):
        doc.metadata["chunk_id"] = idx + 1
        doc.metadata["chunk_length"] = len(doc.page_content)
        
    return chunked_docs
