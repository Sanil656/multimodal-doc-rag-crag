"""
CLI Benchmark Evaluation Runner
================================
Runs automated RAG Triad evaluation benchmarks across test documents and outputs scorecard tables.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from rag_engine.text_splitter import split_documents_into_chunks
from rag_engine.vector_store import get_embedding_function, build_vector_store, get_retriever
from rag_engine.chain import get_llm
from rag_engine.evaluator import evaluate_rag_triad, generate_synthetic_benchmark_dataset
from rag_engine.crag_graph import create_crag_graph


def run_benchmark():
    print("\n" + "=" * 60)
    print("📊 RUNNING AUTOMATED RAG TRIAD BENCHMARK SUITE")
    print("=" * 60)

    # 1. Create sample test documents
    docs = [
        Document(
            page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain by allowing cyclical graphs and state management. LangGraph coordinates nodes like retrieve, grade, and generate.",
            metadata={"source": "langgraph_overview.pdf", "page": 1},
        ),
        Document(
            page_content="Corrective RAG (CRAG) evaluates retrieved documents for relevance before generation. If documents are irrelevant, CRAG rewrites the query for secondary search, reducing hallucinations by 80%.",
            metadata={"source": "crag_architecture.pdf", "page": 2},
        ),
        Document(
            page_content="Token-level contextual pruning removes conversational fluff and redundant sentences from retrieved chunks. This reduces prompt token overhead by 40% to 70%, slashing API costs and prefill latency.",
            metadata={"source": "cost_optimization.pdf", "page": 3},
        ),
    ]

    print(f"📄 Loaded {len(docs)} benchmark test passages.")

    # 2. Vector indexing
    embeddings = get_embedding_function(provider="huggingface")
    vector_store = build_vector_store(docs, embeddings)
    retriever = get_retriever(vector_store, k=3)
    print("✅ Vector index built successfully.")

    # 3. Test queries
    test_queries = [
        "What is the main benefit of Corrective RAG (CRAG)?",
        "How does token pruning reduce API cost in RAG?",
        "What does LangGraph coordinate in stateful workflows?",
    ]

    # Initialize evaluator
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GOOGLE_API_KEY")
    provider = "groq" if os.getenv("GROQ_API_KEY") else ("gemini" if os.getenv("GOOGLE_API_KEY") else "ollama")
    model = "llama-3.1-8b-instant" if provider == "groq" else ("gemini-1.5-flash" if provider == "gemini" else "llama3")

    print(f"🧠 Using Evaluator: {provider} ({model})")
    eval_llm = get_llm(provider=provider, model_name=model, api_key=api_key)

    total_overall = 0.0
    results = []

    for idx, query in enumerate(test_queries):
        print(f"\n--- [Test #{idx+1}] Query: '{query}' ---")
        retrieved = retriever.invoke(query)
        context_str = "\n".join([d.page_content for d in retrieved])
        
        # Generate answer
        gen_prompt = f"Answer this question strictly using the context:\nContext:\n{context_str}\n\nQuestion: {query}"
        generation = eval_llm.invoke(gen_prompt).content

        # Evaluate RAG Triad
        report = evaluate_rag_triad(
            question=query,
            generation=generation,
            context_docs=retrieved,
            evaluator_llm=eval_llm,
        )

        total_overall += report["overall_score"]
        results.append({
            "query": query,
            "context_relevance": report["context_relevance"]["score"],
            "faithfulness": report["faithfulness"]["score"],
            "answer_relevance": report["answer_relevance"]["score"],
            "overall": report["overall_score"],
            "status": report["status"],
        })

        print(f"  • Context Relevance: {report['context_relevance']['score']}/100")
        print(f"  • Faithfulness:       {report['faithfulness']['score']}/100")
        print(f"  • Answer Relevance:  {report['answer_relevance']['score']}/100")
        print(f"  • Composite Score:   {report['overall_score']}/100 [{report['status']}]")

    avg_score = round(total_overall / len(test_queries), 1)
    print("\n" + "=" * 60)
    print(f"🏆 BENCHMARK COMPLETE — Average Composite Score: {avg_score}/100")
    print("=" * 60)
    assert avg_score >= 70.0, "Benchmark composite score should meet quality threshold."
    print("\n[OK] RAG Triad Evaluation Suite PASSED!")


if __name__ == "__main__":
    run_benchmark()
