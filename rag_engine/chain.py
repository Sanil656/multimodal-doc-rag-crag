import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI


RAG_SYSTEM_PROMPT = """You are an expert AI document research assistant. Your task is to accurately and comprehensively answer the user's questions based ONLY on the provided context excerpts from the uploaded document.

CRITICAL GUIDELINES:
1. Grounding: Answer strictly using the information available in the provided document context. Do not invent facts or extrapolate beyond what is stated.
2. Citations: Always reference the specific page number(s) (e.g., "[Page 2]", "[Pages 4-5]") when citing facts or information from the document.
3. Unmentioned Information: If the answer cannot be found in the provided context, state clearly: "Based on the provided document, I cannot find information regarding this question."
4. Tone & Clarity: Provide clear, well-structured, and helpful answers using markdown (bullet points, bold text, or tables where appropriate).

Context from document:
{context}
"""


def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Format retrieved documents into a structured string containing page and source tags.
    """
    formatted_parts = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Document")
        chunk_id = doc.metadata.get("chunk_id", "")
        header = f"--- [Source: {source} | Page: {page} | Chunk ID: {chunk_id}] ---"
        formatted_parts.append(f"{header}\n{doc.page_content}\n")
    return "\n".join(formatted_parts)


def format_chat_history(messages: List[Dict[str, str]]) -> List[tuple]:
    """
    Convert chat messages to LangChain format (role, content).
    """
    history = []
    for msg in messages:
        role = "human" if msg["role"] == "user" else "ai"
        history.append((role, msg["content"]))
    return history


def get_llm(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Initialize the LLM based on provider (gemini or ollama)."""
    if provider == "gemini":
        model_name = model_name or "gemini-1.5-flash"
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API Key is required for Gemini model. "
                "Please provide it in the UI or set GOOGLE_API_KEY in your .env file."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name or "llama3",
            temperature=temperature,
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_rag_chain(
    retriever,
    llm: Optional[ChatGoogleGenerativeAI] = None,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
):
    """
    Build a conversational RAG chain that retrieves context and streams answers with source citation support.
    """
    if llm is None:
        llm = get_llm(model_name=model_name, temperature=temperature, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Context retrieval + answer generation pipeline
    rag_chain = (
        RunnableParallel(
            context=lambda x: format_docs_with_metadata(retriever.invoke(x["question"])),
            raw_docs=lambda x: retriever.invoke(x["question"]),
            question=lambda x: x["question"],
            chat_history=lambda x: x.get("chat_history", []),
        )
        | RunnableParallel(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: x["context"]
                )
                | prompt
                | llm
                | StrOutputParser()
            ),
            source_documents=lambda x: x["raw_docs"],
        )
    )

    return rag_chain
