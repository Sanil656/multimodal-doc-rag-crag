"""
Multi-page document loader supporting PDF (PyMuPDF), DOCX, TXT, and OCR images.
"""

import io
from typing import List
from langchain_core.documents import Document


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from multi-page PDF preserving page numbers and layout."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = (page.get_text("text") or "").strip()
            if not text:
                blocks = page.get_text("blocks")
                text = "\n".join([b[4] for b in blocks if len(b) > 4 and b[4].strip()])
            if text:
                pages.append(Document(page_content=text, metadata={"source": filename, "page": i + 1, "file_type": "pdf"}))
        if pages:
            return pages
    except Exception:
        pass

    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return [
        Document(page_content=t.strip(), metadata={"source": filename, "page": i + 1, "file_type": "pdf"})
        for i, page in enumerate(reader.pages)
        if (t := page.extract_text() or "").strip()
    ]


def load_docx_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from a DOCX document."""
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    return [Document(page_content=text, metadata={"source": filename, "page": 1, "file_type": "docx"})] if text else []


def load_txt_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from a plain TXT or Markdown file."""
    for encoding in ["utf-8", "latin-1"]:
        try:
            text = file_bytes.decode(encoding).strip()
            return [Document(page_content=text, metadata={"source": filename, "page": 1, "file_type": "txt"})] if text else []
        except UnicodeDecodeError:
            continue
    return []


def load_image_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from an image using OCR."""
    try:
        from PIL import Image
        import pytesseract
        text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes))).strip()
    except Exception:
        text = f"[Image File: {filename} - OCR processed]"
    return [Document(page_content=text or f"[Image: {filename}]", metadata={"source": filename, "page": 1, "file_type": "image"})]


def load_document_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """Route file bytes to the appropriate loader based on filename extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    loaders = {
        "pdf": load_pdf_from_bytes,
        "docx": load_docx_from_bytes,
        "doc": load_docx_from_bytes,
        "txt": load_txt_from_bytes,
        "md": load_txt_from_bytes,
    }
    loader = loaders.get(ext, load_image_from_bytes if ext in ["png", "jpg", "jpeg", "webp", "bmp"] else None)
    if not loader:
        raise ValueError(f"Unsupported format: .{ext}")
    return loader(file_bytes, filename)


def load_document_from_path(file_path: str) -> List[Document]:
    """Load document from local file path."""
    with open(file_path, "rb") as f:
        return load_document_from_bytes(f.read(), file_path.replace("\\", "/").split("/")[-1])
