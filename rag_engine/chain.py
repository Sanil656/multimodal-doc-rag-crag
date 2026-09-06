"""
LLM Factory & Conversational Chain definitions for Groq, Ollama, and Gemini.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

RAG_SYSTEM_PROMPT = """You are an expert AI document research assistant. Answer accurately based ONLY on the provided context excerpts.
1. Grounding: Answer strictly using the document context. Never invent facts.
2. Citations: Reference specific page numbers (e.g. "[Page 2]").
3. Unmentioned Information: If not in context, state clearly: "Based on the provided document, I cannot find information regarding this question."

Context:
{context}"""


def format_docs_with_metadata(docs: List[Document]) -> str:
    """Format retrieved documents with source and page tags."""
    return "\n\n".join(
        f"--- [Source: {d.metadata.get('source', 'Doc')} | Page: {d.metadata.get('page', 1)}] ---\n{d.page_content}"
        for d in docs
    )


def format_chat_history(messages: List[Dict[str, str]]) -> List[tuple]:
    """Convert UI chat messages to LangChain tuples."""
    return [("human" if m["role"] == "user" else "ai", m["content"]) for m in messages]


def get_llm(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Factory creating LLM instances for Gemini, Groq, or local Ollama."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API Key is required.")
        return ChatGoogleGenerativeAI(model=model_name or "gemini-1.5-flash", temperature=temperature, google_api_key=key)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API Key is required.")
        return ChatGroq(model_name=model_name or "llama-3.3-70b-versatile", temperature=temperature, groq_api_key=key)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name or "llama3", temperature=temperature, base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    raise ValueError(f"Unsupported LLM provider: {provider}")


def create_rag_chain(retriever: Any, llm: Any):
    """Build standard RAG retrieval QA chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return (
        RunnableParallel(
            context=lambda x: format_docs_with_metadata(retriever.invoke(x["question"])),
            raw_docs=lambda x: retriever.invoke(x["question"]),
            question=lambda x: x["question"],
            chat_history=lambda x: x.get("chat_history", []),
        )
        | RunnableParallel(
            answer=(RunnablePassthrough.assign(context=lambda x: x["context"]) | prompt | llm | StrOutputParser()),
            source_documents=lambda x: x["raw_docs"],
        )
    )
