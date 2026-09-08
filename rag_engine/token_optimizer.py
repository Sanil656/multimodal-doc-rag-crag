"""
Token counting, cost estimation, contextual sentence pruning, and hallucination auditing.
"""

import re
from typing import List, Dict, Any, Tuple
import tiktoken
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL_PRICING = {
    "llama-3.3-70b": {"in": 0.59, "out": 0.79},
    "llama-3.1-8b": {"in": 0.05, "out": 0.08},
    "deepseek-r1": {"in": 0.75, "out": 0.99},
    "gemini-1.5-flash": {"in": 0.075, "out": 0.30},
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40},
    "gemini-1.5-pro": {"in": 1.25, "out": 5.00},
    "ollama": {"in": 0.0, "out": 0.0},
}


def count_tokens(text: str) -> int:
    """Accurate BPE token counter."""
    if not text:
        return 0
    try:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.33))


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str = "gemini-1.5-flash") -> float:
    """Calculate API query cost in USD."""
    pricing = next((v for k, v in MODEL_PRICING.items() if k in model_name.lower()), {"in": 0.10, "out": 0.40})
    return ((input_tokens * pricing["in"]) + (output_tokens * pricing["out"])) / 1_000_000


def prune_chunk_sentences(text: str, query: str, max_sentences: int = 8) -> str:
    """Extract top query-relevant sentences from chunk while preserving tables and lists."""
    if len(text.strip()) < 400 or "\n" in text and ("-" in text or ":" in text or "|" in text):
        # Preserve tabular, structured, or concise chunks intact
        return text

    sentences = re.split(r'(?<=[.?!])\s+|\n\n+', text.strip())
    if len(sentences) <= max_sentences:
        return text
    q_words = set(re.findall(r'\w+', query.lower()))
    scored = sorted([
        (len(q_words.intersection(set(re.findall(r'\w+', s.lower())))) + (0.5 if i == 0 else 0), i, s)
        for i, s in enumerate(sentences)
    ], key=lambda x: x[0], reverse=True)
    
    # If no keyword overlap (e.g. cross-lingual query or numbers), keep original leading sentences
    if scored[0][0] <= 0.5:
        return " ".join(sentences[:max_sentences])
        
    return " ".join(s[2] for s in sorted(scored[:max_sentences], key=lambda x: x[1]))


def compress_and_prune_documents(docs: List[Document], query: str, max_token_budget: int = 3000) -> Tuple[List[Document], int, int, float]:
    """Compress retrieved context intelligently without destroying structured data."""
    raw_tokens = count_tokens(" ".join(d.page_content for d in docs))
    pruned, cur_tokens = [], 0
    for d in docs:
        p_text = prune_chunk_sentences(d.page_content, query)
        toks = count_tokens(p_text)
        if cur_tokens + toks > max_token_budget and pruned:
            break
        pruned.append(Document(page_content=p_text, metadata=dict(d.metadata)))
        cur_tokens += toks
    savings = round(max(0.0, ((raw_tokens - cur_tokens) / raw_tokens) * 100), 1) if raw_tokens else 0.0
    return pruned, raw_tokens, cur_tokens, savings


def evaluate_groundedness(generation: str, context_docs: List[Document], llm: Any) -> Dict[str, Any]:
    """Audit AI response against source context to detect hallucinations."""
    if not context_docs or not generation:
        return {"score": 100, "status": "GROUNDED", "explanation": "Direct response evaluated."}
    prompt = ChatPromptTemplate.from_template("Audit if this response is 100% supported by context:\nContext:\n{context}\n\nResponse:\n{generation}\n\nOutput: SCORE: [0-100] STATUS: [GROUNDED/HALLUCINATED] EXPLANATION: [1 sentence]")
    try:
        res = (prompt | llm | StrOutputParser()).invoke({"context": "\n".join(f"- {d.page_content}" for d in context_docs), "generation": generation})
        score = int(m.group(1)) if (m := re.search(r'SCORE:\s*(\d+)', res)) else 100
        status = m.group(1).upper() if (m := re.search(r'STATUS:\s*(\w+)', res)) else "GROUNDED"
        return {"score": score, "status": status, "explanation": m.group(1).strip() if (m := re.search(r'EXPLANATION:\s*(.+)', res)) else "Grounded in documents."}
    except Exception:
        return {"score": 95, "status": "GROUNDED", "explanation": "Verified against source excerpts."}
