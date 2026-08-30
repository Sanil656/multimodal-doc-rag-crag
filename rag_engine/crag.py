import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from .chain import format_docs_with_metadata, RAG_SYSTEM_PROMPT, get_llm


# ---------------- PROMPTS FOR CRAG WORKFLOW ----------------

GRADER_PROMPT = """You are a strict document evaluator assessing whether a retrieved document snippet is relevant to the user's question.

Question: {question}

Document Snippet:
{document}

Evaluate if the snippet contains information, context, or facts that help answer the question.
Respond with ONLY ONE WORD: "yes" if relevant, or "no" if irrelevant. Do not provide any other text.
"""

REWRITER_PROMPT = """You are an AI assistant optimizing search queries for a vector database retrieval system.
Look at the user's original question and conversation history, and formulate an improved, keyword-rich search query that will better retrieve relevant passages from a document.

Conversation History:
{chat_history}

Original Question:
{question}

Provide ONLY the reformulated search query, without explanations or quotes.
"""


# ---------------- CRAG CORE LOGIC ----------------

def grade_document_relevance(
    doc: Document,
    question: str,
    llm: Any,
) -> bool:
    """
    Grades a single document chunk for relevance against the question.
    Returns True if relevant ('yes'), False otherwise.
    """
    grader_chain = (
        ChatPromptTemplate.from_template(GRADER_PROMPT)
        | llm
        | StrOutputParser()
    )
    try:
        response = grader_chain.invoke({"question": question, "document": doc.page_content}).strip().lower()
        return "yes" in response
    except Exception:
        # Fallback to including chunk if grading call encounters an error
        return True


def rewrite_query(
    question: str,
    chat_history_str: str,
    llm: Any,
) -> str:
    """
    Reformulates the user's query into an optimized retrieval query.
    """
    rewriter_chain = (
        ChatPromptTemplate.from_template(REWRITER_PROMPT)
        | llm
        | StrOutputParser()
    )
    try:
        improved_query = rewriter_chain.invoke({
            "question": question,
            "chat_history": chat_history_str,
        }).strip()
        return improved_query if improved_query else question
    except Exception:
        return question


def run_crag_pipeline(
    question: str,
    retriever,
    llm: Optional[Any] = None,
    chat_history: Optional[List[tuple]] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the Corrective RAG (CRAG) pipeline:
    1. Retrieve initial candidate chunks from Vector Store
    2. Grade retrieved chunks for relevance
    3. If relevance is low, rewrite query and perform secondary retrieval
    4. Filter and refine context strips
    5. Generate answer strictly grounded on verified relevant chunks
    """
    if llm is None:
        llm = get_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    chat_history = chat_history or []
    chat_history_str = "\n".join([f"{role}: {text}" for role, text in chat_history])

    # Step 1: Initial Retrieval
    initial_docs = retriever.invoke(question)
    
    # Step 2: Grade Documents
    relevant_docs = []
    for doc in initial_docs:
        if grade_document_relevance(doc, question, llm):
            relevant_docs.append(doc)

    query_was_rewritten = False
    rewritten_query = question

    # Step 3: Corrective action if no documents were deemed relevant
    if not relevant_docs and len(initial_docs) > 0:
        query_was_rewritten = True
        rewritten_query = rewrite_query(question, chat_history_str, llm)
        
        # Re-retrieve with improved query
        secondary_docs = retriever.invoke(rewritten_query)
        for doc in secondary_docs:
            if grade_document_relevance(doc, question, llm):
                relevant_docs.append(doc)

    # Step 4: Context Refinement / Generation
    if not relevant_docs:
        # If still no relevant documents found across pages
        return {
            "answer": "Based on the provided document, I cannot find sufficient relevant information to answer this question accurately.",
            "source_documents": [],
            "crag_stats": {
                "initial_retrieved": len(initial_docs),
                "relevant_filtered": 0,
                "query_rewritten": query_was_rewritten,
                "rewritten_query": rewritten_query if query_was_rewritten else None,
            },
        }

    # Step 5: Format context and generate grounded answer
    formatted_context = format_docs_with_metadata(relevant_docs)
    
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])
    
    generation_chain = generator_prompt | llm | StrOutputParser()
    answer = generation_chain.invoke({
        "question": question,
        "context": formatted_context,
    })

    return {
        "answer": answer,
        "source_documents": relevant_docs,
        "crag_stats": {
            "initial_retrieved": len(initial_docs),
            "relevant_filtered": len(relevant_docs),
            "query_rewritten": query_was_rewritten,
            "rewritten_query": rewritten_query if query_was_rewritten else None,
        },
    }


def stream_crag_pipeline(
    question: str,
    retriever,
    llm: Optional[Any] = None,
    chat_history: Optional[List[tuple]] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Executes the CRAG pipeline and yields tokens as a stream for real-time UI rendering.
    Returns: (token_generator, relevant_docs, crag_stats)
    """
    if llm is None:
        llm = get_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    chat_history = chat_history or []
    chat_history_str = "\n".join([f"{role}: {text}" for role, text in chat_history])

    # 1. Retrieve
    initial_docs = retriever.invoke(question)
    
    # 2. Grade
    relevant_docs = []
    for doc in initial_docs:
        if grade_document_relevance(doc, question, llm):
            relevant_docs.append(doc)

    query_was_rewritten = False
    rewritten_query = question

    # 3. Rewrite if needed
    if not relevant_docs and len(initial_docs) > 0:
        query_was_rewritten = True
        rewritten_query = rewrite_query(question, chat_history_str, llm)
        secondary_docs = retriever.invoke(rewritten_query)
        for doc in secondary_docs:
            if grade_document_relevance(doc, question, llm):
                relevant_docs.append(doc)

    crag_stats = {
        "initial_retrieved": len(initial_docs),
        "relevant_filtered": len(relevant_docs),
        "query_rewritten": query_was_rewritten,
        "rewritten_query": rewritten_query if query_was_rewritten else None,
    }

    if not relevant_docs:
        def fallback_generator():
            yield "Based on the provided document, I cannot find sufficient relevant information to answer this question accurately."
        return fallback_generator(), [], crag_stats

    # 4. Stream generator
    formatted_context = format_docs_with_metadata(relevant_docs)
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])
    
    generation_chain = generator_prompt | llm | StrOutputParser()
    token_stream = generation_chain.stream({
        "question": question,
        "context": formatted_context,
    })

    return token_stream, relevant_docs, crag_stats
