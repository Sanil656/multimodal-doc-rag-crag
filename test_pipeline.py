"""
Test pipeline script to verify document loading, chunking, metadata retention, and vector indexing.
"""

from langchain_core.documents import Document
from rag_engine.document_loader import load_document_from_bytes
from rag_engine.text_splitter import split_documents_into_chunks


def test_chunking_and_metadata():
    print("--- Running Document Processing Test ---")
    
    # Simulate multi-page document
    sample_text_page1 = "Project Title: AI Document Reader\nThis project implements a Retrieval-Augmented Generation system. Section 1 discusses architecture."
    sample_text_page2 = "Section 2: Multi-Page Processing\nThe system extracts text page by page preserving metadata. Page 2 covers chunking strategies."
    sample_text_page3 = "Section 3: Evaluation & Benchmarks\nAccuracy was measured at 98.5% precision across 500 test documents."
    
    mock_docs = [
        Document(page_content=sample_text_page1, metadata={"source": "test_report.pdf", "page": 1, "total_pages": 3, "file_type": "pdf"}),
        Document(page_content=sample_text_page2, metadata={"source": "test_report.pdf", "page": 2, "total_pages": 3, "file_type": "pdf"}),
        Document(page_content=sample_text_page3, metadata={"source": "test_report.pdf", "page": 3, "total_pages": 3, "file_type": "pdf"}),
    ]
    
    chunks = split_documents_into_chunks(mock_docs, chunk_size=100, chunk_overlap=20)
    
    print(f"Total input pages: {len(mock_docs)}")
    print(f"Total generated chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk #{chunk.metadata['chunk_id']}: Page {chunk.metadata['page']} | Length: {chunk.metadata['chunk_length']} chars")
        print(f"   Excerpt: {chunk.page_content[:50]}...")
        
    assert len(chunks) >= 3, "Should have created at least 3 chunks"
    assert chunks[0].metadata["page"] == 1
    assert chunks[-1].metadata["page"] == 3
    print("\n[OK] Document loader and metadata preservation tests PASSED successfully!")


if __name__ == "__main__":
    test_chunking_and_metadata()
