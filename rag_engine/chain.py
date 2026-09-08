"""
LLM Factory & Conversational Chain definitions for Groq, Ollama, and Gemini.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

RAG_SYSTEM_PROMPT = """You are a universal AI document research, analysis, and comparison intelligence assistant. Answer accurately using the provided context excerpts (Document Chunks and/or Web Search results).

Universal Response Protocols:
1. Direct Answer First: ALWAYS start with the direct, concise answer on the very first line:
   - Verification / Binary Queries: Start immediately with "**Yes.**" or "**No.**" followed by the core fact.
   - Values, Entities, Stats, Formulas & Definitions: Provide the exact value or definition upfront without conversational filler.
   - Conversions, Tables, Code, Lists, or Translations: Deliver the requested extracted list, table, or conversion directly.
   - Summaries / Open Questions: Provide a crisp 1-sentence core takeaway first.
2. Supporting Context & Exact Citations:
   - Provide concise bullet points or structured explanations supporting the answer.
   - Cite exact document source pages (e.g., `[Doc: filename | Page X]`) or web references (e.g., `[Web: Title]`).
3. Cross-Source Verification (Doc vs. Web):
   - When both document context and live web search are available, synthesize both sources and clearly highlight any agreements, updates, or discrepancies.
4. Strict Grounding & Zero Hallucination:
   - Base all claims strictly on the provided context. If a detail is missing from both documents and web search, state clearly what could and could not be found.

Context:
{context}"""


def format_docs_with_metadata(docs: List[Document]) -> str:
    """Format retrieved documents with source and page tags."""
    return "\n\n".join(f"--- [Source: {d.metadata.get('source', 'Doc')} | Page: {d.metadata.get('page', 1)}] ---\n{d.page_content}" for d in docs)


def format_chat_history(messages: List[Dict[str, str]]) -> List[tuple]:
    """Convert UI chat messages to LangChain tuples."""
    return [("human" if m["role"] == "user" else "ai", m["content"]) for m in messages]


def get_llm(provider: str = "gemini", model_name: Optional[str] = None, temperature: float = 0.2, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Factory creating LLM instances for Gemini, Groq, or local Ollama."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key: raise ValueError("Google API Key is required.")
        return ChatGoogleGenerativeAI(model=model_name or "gemini-1.5-flash", temperature=temperature, google_api_key=key)

    if provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key: raise ValueError("Groq API Key is required.")
        return ChatGroq(model_name=model_name or "llama-3.3-70b-versatile", temperature=temperature, groq_api_key=key)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name or "llama3", temperature=temperature, base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    raise ValueError(f"Unsupported LLM provider: {provider}")


def create_rag_chain(retriever: Any, llm: Any):
    """Build standard RAG retrieval QA chain."""
    prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")])
    return (
        RunnableParallel(context=lambda x: format_docs_with_metadata(retriever.invoke(x["question"])), raw_docs=lambda x: retriever.invoke(x["question"]), question=lambda x: x["question"], chat_history=lambda x: x.get("chat_history", []))
        | RunnableParallel(answer=(RunnablePassthrough.assign(context=lambda x: x["context"]) | prompt | llm | StrOutputParser()), source_documents=lambda x: x["raw_docs"])
    )
