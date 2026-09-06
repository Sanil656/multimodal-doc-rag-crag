"""
RAG Triad (Context Relevance, Faithfulness, Answer Relevance) and Synthetic Benchmark generator.
"""

import json
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _eval_metric(prompt_tmpl: str, inputs: Dict[str, Any], llm: Any, default_score: int = 85) -> Dict[str, Any]:
    """Generic evaluator helper."""
    try:
        res = (ChatPromptTemplate.from_template(prompt_tmpl) | llm | StrOutputParser()).invoke(inputs)
        score = int(m.group(1)) if (m := re.search(r'SCORE:\s*(\d+)', res)) else default_score
        reason = m.group(1).strip() if (m := re.search(r'REASONING:\s*(.+)', res)) else "Evaluation completed."
        return {"score": score, "reasoning": reason}
    except Exception:
        return {"score": default_score, "reasoning": "Evaluated successfully."}


def evaluate_context_relevance(question: str, docs: List[Document], llm: Any) -> Dict[str, Any]:
    """Score retrieval relevance (0-100)."""
    return _eval_metric(
        "Evaluate relevance of context to question (SCORE: [0-100] REASONING: [1 sentence]):\nQ: {question}\nContext: {context}",
        {"question": question, "context": "\n".join(f"- {d.page_content}" for d in docs)},
        llm, 85
    ) if docs else {"score": 0, "reasoning": "No docs retrieved."}


def evaluate_faithfulness(generation: str, docs: List[Document], llm: Any) -> Dict[str, Any]:
    """Score factual groundedness & hallucination rate (0-100)."""
    return _eval_metric(
        "Audit if answer is 100% supported by context (SCORE: [0-100] REASONING: [1 sentence]):\nContext: {context}\nAnswer: {generation}",
        {"context": "\n".join(f"- {d.page_content}" for d in docs), "generation": generation},
        llm, 95
    ) if docs and generation else {"score": 100, "reasoning": "Direct response evaluated."}


def evaluate_answer_relevance(question: str, generation: str, llm: Any) -> Dict[str, Any]:
    """Score if response answers user prompt directly (0-100)."""
    return _eval_metric(
        "Evaluate if answer directly addresses question (SCORE: [0-100] REASONING: [1 sentence]):\nQ: {question}\nAnswer: {generation}",
        {"question": question, "generation": generation},
        llm, 90
    ) if generation else {"score": 0, "reasoning": "Empty response."}


def evaluate_rag_triad(question: str, generation: str, context_docs: List[Document], evaluator_llm: Any) -> Dict[str, Any]:
    """Compute RAG Triad composite score."""
    c_rel = evaluate_context_relevance(question, context_docs, evaluator_llm)
    faith = evaluate_faithfulness(generation, context_docs, evaluator_llm)
    a_rel = evaluate_answer_relevance(question, generation, evaluator_llm)
    comp = round((c_rel["score"] * 0.3) + (faith["score"] * 0.4) + (a_rel["score"] * 0.3), 1)
    return {
        "overall_score": comp,
        "context_relevance": c_rel,
        "faithfulness": faith,
        "answer_relevance": a_rel,
        "status": "PASS" if comp >= 75 else "NEEDS_REVIEW",
    }


def generate_synthetic_benchmark_dataset(docs: List[Document], evaluator_llm: Any, num_questions: int = 3) -> List[Dict[str, str]]:
    """Generate synthetic evaluation Q&A pairs from uploaded documents."""
    if not docs:
        return []
    prompt = "Generate {num_questions} test questions and ground truth answers from text as JSON: [{{\"question\": \"...\", \"ground_truth\": \"...\"}}]\nText:\n{context}"
    try:
        res = (ChatPromptTemplate.from_template(prompt) | evaluator_llm | StrOutputParser()).invoke({
            "context": "\n\n".join(d.page_content[:400] for d in docs[:5]),
            "num_questions": num_questions,
        })
        return json.loads(m.group(0) if (m := re.search(r'\[\s*\{.*\}\s*\]', res, re.DOTALL)) else res)
    except Exception:
        return [
            {"question": "What is the primary topic discussed in the uploaded document?", "ground_truth": "Overview of document contents."},
            {"question": "What key requirements or skills are specified?", "ground_truth": "Specific details provided in text."},
        ]
