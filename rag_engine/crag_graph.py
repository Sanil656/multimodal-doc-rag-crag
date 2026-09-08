"""
High-Speed Stateful LangGraph CRAG workflow with Batch-Parallel Grading, Web Search, and Context Pruning.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from .chain import format_docs_with_metadata, RAG_SYSTEM_PROMPT, get_llm
from .token_optimizer import count_tokens, calculate_cost, compress_and_prune_documents, evaluate_groundedness


class GraphState(TypedDict):
    question: str
    chat_history: List[tuple]
    documents: List[Document]
    generation: str
    query_rewritten: bool
    rewritten_query: str
    initial_retrieved: int
    relevant_filtered: int
    raw_tokens: int
    pruned_tokens: int
    token_savings_pct: float
    groundedness_score: int
    hallucination_status: str
    hallucination_explanation: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    web_search_used: bool


def perform_web_search(query: str, max_results: int = 3) -> List[Document]:
    """Execute live DuckDuckGo web search without requiring external API keys."""
    try:
        from duckduckgo_search import DDGS
        return [
            Document(page_content=f"Title: {r.get('title', 'Web Source')}\nSnippet: {r.get('body', '')}", metadata={"source": f"Web: {r.get('title', 'Web')}", "url": r.get("href", ""), "page": "Web"})
            for r in DDGS().text(query, max_results=max_results)
            if r.get("body")
        ]
    except Exception:
        return []


def create_crag_graph(retriever: Any, llm: Any, evaluator_llm: Optional[Any] = None, model_name: str = "gemini-1.5-flash", enable_web_search: bool = True):
    """Build and compile the high-speed LangGraph CRAG state machine with Web Search comparison."""
    eval_llm = evaluator_llm or llm

    def retrieve_node(state: GraphState) -> Dict[str, Any]:
        docs = retriever.invoke(state.get("rewritten_query") or state["question"])
        return {"documents": docs, "initial_retrieved": len(docs)}

    def grade_and_prune_node(state: GraphState) -> Dict[str, Any]:
        q, docs = state["question"], state.get("documents", [])
        if not docs:
            return {"documents": [], "relevant_filtered": 0, "raw_tokens": 0, "pruned_tokens": 0, "token_savings_pct": 0.0}

        batch_context = "\n\n".join(f"[{i+1}] {d.page_content[:600]}" for i, d in enumerate(docs))
        batch_prompt = "Evaluate which document chunks contain facts, vocabulary, tables, numbers, or context relevant to answering the question.\nQuestion: {question}\n\nDocument Chunks:\n{batch_context}\n\nRespond with ONLY comma-separated numbers (e.g., '1, 2, 4' or 'ALL' or 'NONE'):"
        
        try:
            res = (ChatPromptTemplate.from_template(batch_prompt) | eval_llm | StrOutputParser()).invoke({"question": q, "batch_context": batch_context}).strip()
            if "none" in res.lower() and len(res.split()) <= 2:
                relevant = docs[:2] if state.get("query_rewritten") else []
            elif "all" in res.lower() or not re.findall(r'\d+', res):
                relevant = docs
            else:
                indices = [int(n) - 1 for n in re.findall(r'\d+', res) if 0 <= int(n) - 1 < len(docs)]
                relevant = [docs[i] for i in indices] if indices else docs
        except Exception:
            relevant = docs

        pruned, raw_t, pruned_t, savings = compress_and_prune_documents(relevant, query=q) if relevant else ([], 0, 0, 0.0)
        return {"documents": pruned, "relevant_filtered": len(pruned), "raw_tokens": raw_t, "pruned_tokens": pruned_t, "token_savings_pct": savings}

    def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
        hist = "\n".join(f"{r}: {c}" for r, c in state.get("chat_history", []))
        try:
            nq = (ChatPromptTemplate.from_template("Rewrite search query to optimize vector retrieval:\nHistory: {history}\nQuestion: {question}\nOptimized Query:") | eval_llm | StrOutputParser()).invoke({"question": state["question"], "history": hist}).strip()
        except Exception:
            nq = state["question"]
        return {"rewritten_query": nq, "query_rewritten": True}

    def web_search_node(state: GraphState) -> Dict[str, Any]:
        search_query = state.get("rewritten_query") or state["question"]
        web_docs = perform_web_search(search_query, max_results=3)
        return {"documents": list(state.get("documents", [])) + web_docs, "web_search_used": True}

    def generate_node(state: GraphState) -> Dict[str, Any]:
        q, docs, hist = state["question"], state.get("documents", []), state.get("chat_history", [])
        c_str = format_docs_with_metadata(docs) if docs else "No relevant document context found."
        ans = (ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{question}")]) | llm | StrOutputParser()).invoke({"context": c_str, "chat_history": hist, "question": q})
        in_t, out_t = count_tokens(c_str) + count_tokens(q), count_tokens(ans)
        return {"generation": ans, "input_tokens": in_t, "output_tokens": out_t, "estimated_cost_usd": calculate_cost(in_t, out_t, model_name)}

    def hallucination_guard_node(state: GraphState) -> Dict[str, Any]:
        report = evaluate_groundedness(state.get("generation", ""), state.get("documents", []), eval_llm)
        return {"groundedness_score": report["score"], "hallucination_status": report["status"], "hallucination_explanation": report["explanation"]}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_and_prune", grade_and_prune_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_guard", hallucination_guard_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_and_prune")
    workflow.add_conditional_edges("grade_and_prune", lambda s: "rewrite_query" if (not s.get("documents") and not s.get("query_rewritten")) else ("web_search" if enable_web_search else "generate"), {"generate": "generate", "rewrite_query": "rewrite_query", "web_search": "web_search"})
    workflow.add_edge("rewrite_query", "web_search" if enable_web_search else "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "hallucination_guard")
    workflow.add_edge("hallucination_guard", END)

    return workflow.compile()


def stream_langgraph_crag_pipeline(question: str, retriever: Any, chat_history: Optional[List[tuple]] = None, provider: str = "gemini", model_name: Optional[str] = None, temperature: float = 0.2, api_key: Optional[str] = None, base_url: Optional[str] = None, enable_web_search: bool = True) -> Tuple[Any, List[Document], Dict[str, Any]]:
    """Execute LangGraph CRAG pipeline with sub-second retrieval, Web Search augmentation, and token streaming."""
    active_m = model_name or ("gemini-1.5-flash" if provider == "gemini" else "openai/gpt-oss-120b")
    gen_llm = get_llm(provider, active_m, temperature, api_key, base_url)
    eval_m = "openai/gpt-oss-20b" if (provider == "groq" and "openai" in active_m) else active_m
    eval_llm = get_llm(provider, eval_m, 0.0, api_key, base_url)

    app = create_crag_graph(retriever, gen_llm, eval_llm, active_m, enable_web_search=enable_web_search)
    init_state = {"question": question, "chat_history": chat_history or [], "documents": [], "generation": "", "query_rewritten": False, "rewritten_query": "", "initial_retrieved": 0, "relevant_filtered": 0, "raw_tokens": 0, "pruned_tokens": 0, "token_savings_pct": 0.0, "groundedness_score": 100, "hallucination_status": "GROUNDED", "hallucination_explanation": "Grounded.", "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "web_search_used": False}

    final_state = app.invoke(init_state)
    docs, in_t, out_t = final_state.get("documents", []), final_state.get("input_tokens", 0), final_state.get("output_tokens", 0)

    stats = {
        "initial_retrieved": final_state.get("initial_retrieved", len(docs)), "relevant_filtered": len(docs),
        "query_rewritten": final_state.get("query_rewritten", False), "rewritten_query": final_state.get("rewritten_query", ""),
        "token_savings_pct": final_state.get("token_savings_pct", 0.0), "groundedness_score": final_state.get("groundedness_score", 100),
        "hallucination_status": final_state.get("hallucination_status", "GROUNDED"), "total_tokens": in_t + out_t,
        "input_tokens": in_t, "output_tokens": out_t, "estimated_cost_usd": final_state.get("estimated_cost_usd", 0.0),
        "web_search_used": final_state.get("web_search_used", False)
    }

    c_str = format_docs_with_metadata(docs) if docs else "No relevant document context found."
    stream = (ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{question}")]) | gen_llm | StrOutputParser()).stream({"context": c_str, "chat_history": chat_history or [], "question": question})

    return stream, docs, stats
