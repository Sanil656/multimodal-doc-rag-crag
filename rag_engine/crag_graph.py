"""
LangGraph Corrective RAG (CRAG) Workflow
=========================================
A stateful, agentic graph implementation of Corrective RAG using LangGraph.

Architecture & State Flow:
--------------------------
[User Query] ──> [Node: retrieve]
                      │
                      ▼
             [Node: grade_documents]
                      │
       ┌──────────────┴──────────────┐
       │ (Relevant Chunks Found)     │ (All Chunks Irrelevant / Low Confidence)
       ▼                             ▼
[Node: generate]              [Node: rewrite_query]
       │                             │
       ▼                             └──> [Node: retrieve] (Secondary Search)
     [END]
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from .chain import format_docs_with_metadata, RAG_SYSTEM_PROMPT, get_llm


# ---------------- GRAPH STATE DEFINITION ----------------
class GraphState(TypedDict):
    """
    Represents the state of our agentic RAG graph.

    Attributes:
        question: User's original or active query
        chat_history: Prior conversational messages
        documents: List of retrieved / filtered document passages
        generation: Final synthesized response from LLM
        query_rewritten: Boolean indicating if query reformulation was triggered
        rewritten_query: The rewritten search query (if any)
        initial_retrieved: Initial chunk count before grading
        relevant_filtered: Number of chunks kept after grading
    """
    question: str
    chat_history: List[tuple]
    documents: List[Document]
    generation: str
    query_rewritten: bool
    rewritten_query: str
    initial_retrieved: int
    relevant_filtered: int


# ---------------- PROMPT TEMPLATES ----------------
GRADER_PROMPT = """You are a strict document evaluator assessing whether a retrieved document snippet is relevant to the user's question.

Question: {question}

Document Snippet:
{document}

Evaluate if the snippet contains information, context, or facts that help answer the question.
Respond with ONLY ONE WORD: "yes" if relevant, or "no" if irrelevant. Do not provide any other text."""

REWRITER_PROMPT = """You are an AI assistant optimizing search queries for a vector database retrieval system.
Look at the user's original question and conversation history, and formulate an improved, keyword-rich search query that will better retrieve relevant passages from a document.

Conversation History:
{chat_history}

Original Question:
{question}

Provide ONLY the reformulated search query, without explanations or quotes."""


# ---------------- GRAPH NODES & HELPERS ----------------

def create_crag_graph(
    retriever: Any,
    llm: Any,
    evaluator_llm: Optional[Any] = None,
):
    """
    Builds and compiles the stateful LangGraph workflow for Corrective RAG.
    """
    eval_llm = evaluator_llm or llm

    # 1. Node: Retrieve Chunks from Vector Store
    def retrieve_node(state: GraphState) -> Dict[str, Any]:
        """Retrieves documents from vector store using active question."""
        query = state.get("rewritten_query") or state["question"]
        docs = retriever.invoke(query)
        return {
            "documents": docs,
            "initial_retrieved": len(docs),
        }

    # 2. Node: Grade Retrieved Documents for Relevance
    def grade_documents_node(state: GraphState) -> Dict[str, Any]:
        """Filters out irrelevant chunks using an internal LLM evaluator."""
        question = state["question"]
        docs = state.get("documents", [])

        grader_chain = ChatPromptTemplate.from_template(GRADER_PROMPT) | eval_llm | StrOutputParser()
        relevant_docs = []

        for doc in docs:
            try:
                score = grader_chain.invoke({"question": question, "document": doc.page_content}).strip().lower()
                if "yes" in score:
                    relevant_docs.append(doc)
            except Exception:
                # On timeout/error, keep chunk to prevent false exclusion
                relevant_docs.append(doc)

        return {
            "documents": relevant_docs,
            "relevant_filtered": len(relevant_docs),
        }

    # 3. Node: Rewrite Query (Triggered when retrieval is noisy)
    def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
        """Reformulates query for refined secondary retrieval."""
        question = state["question"]
        chat_hist = state.get("chat_history", [])
        hist_str = "\n".join([f"{role}: {content}" for role, content in chat_hist]) if chat_hist else "None"

        rewriter_chain = ChatPromptTemplate.from_template(REWRITER_PROMPT) | eval_llm | StrOutputParser()
        try:
            new_query = rewriter_chain.invoke({"question": question, "chat_history": hist_str}).strip()
        except Exception:
            new_query = question

        return {
            "rewritten_query": new_query,
            "query_rewritten": True,
        }

    # 4. Node: Generate Grounded Answer
    def generate_node(state: GraphState) -> Dict[str, Any]:
        """Synthesizes final answer using strictly filtered context."""
        question = state["question"]
        docs = state.get("documents", [])
        chat_history = state.get("chat_history", [])

        context_str = format_docs_with_metadata(docs) if docs else "No relevant document excerpts found."

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        generation_chain = prompt | llm | StrOutputParser()
        answer = generation_chain.invoke({
            "context": context_str,
            "chat_history": chat_history,
            "question": question,
        })

        return {"generation": answer}

    # Conditional Router Edge: Decide whether to generate or rewrite query
    def decide_to_generate(state: GraphState) -> Literal["generate", "rewrite_query"]:
        """
        Determines whether filtered documents are sufficient for generation
        or if query needs to be rewritten.
        """
        relevant_docs = state.get("documents", [])
        already_rewritten = state.get("query_rewritten", False)

        # If we have relevant documents OR we already retried once, generate answer
        if len(relevant_docs) > 0 or already_rewritten:
            return "generate"
        else:
            return "rewrite_query"

    # ---------------- BUILD GRAPH ----------------
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)

    # Add Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ---------------- STREAMING PIPELINE INTERFACE ----------------

def stream_langgraph_crag_pipeline(
    question: str,
    retriever: Any,
    chat_history: Optional[List[tuple]] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[Any, List[Document], Dict[str, Any]]:
    """
    Executes the LangGraph CRAG workflow and returns real-time streaming tokens.
    Returns: (token_stream_generator, source_documents_list, crag_stats_dict)
    """
    chat_history = chat_history or []

    # Generator LLM
    generator_llm = get_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Lightweight Evaluator LLM
    eval_model = "gemini-1.5-flash" if provider == "gemini" else ("llama-3.1-8b-instant" if provider == "groq" else model_name)
    evaluator_llm = get_llm(
        provider=provider,
        model_name=eval_model,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    # Compile LangGraph
    crag_app = create_crag_graph(
        retriever=retriever,
        llm=generator_llm,
        evaluator_llm=evaluator_llm,
    )

    # Initial State
    initial_state = {
        "question": question,
        "chat_history": chat_history,
        "documents": [],
        "generation": "",
        "query_rewritten": False,
        "rewritten_query": "",
        "initial_retrieved": 0,
        "relevant_filtered": 0,
    }

    # Execute graph up to the final generation node
    # Step 1: Run retrieval, grading, and potential query rewriting
    final_state = crag_app.invoke(initial_state)

    source_docs = final_state.get("documents", [])
    crag_stats = {
        "initial_retrieved": final_state.get("initial_retrieved", len(source_docs)),
        "relevant_filtered": final_state.get("relevant_filtered", len(source_docs)),
        "query_rewritten": final_state.get("query_rewritten", False),
        "rewritten_query": final_state.get("rewritten_query", ""),
    }

    # Step 2: Stream tokens from the final prompt
    context_str = format_docs_with_metadata(source_docs) if source_docs else "No relevant document excerpts found."
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    streaming_chain = prompt | generator_llm | StrOutputParser()

    token_stream = streaming_chain.stream({
        "context": context_str,
        "chat_history": chat_history,
        "question": question,
    })

    return token_stream, source_docs, crag_stats
