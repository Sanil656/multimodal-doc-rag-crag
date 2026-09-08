import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_engine.document_loader import load_document_from_bytes
from rag_engine.text_splitter import split_documents_into_chunks
from rag_engine.vector_store import get_embedding_function, build_vector_store, get_retriever, clear_persisted_vector_store
from rag_engine.chain import create_rag_chain, format_chat_history, get_llm
from rag_engine.crag_graph import stream_langgraph_crag_pipeline
from rag_engine.evaluator import evaluate_rag_triad, generate_synthetic_benchmark_dataset

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(page_title="DocuQuery AI - Multi-Document RAG & Evaluation", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.1rem; }
    .doc-banner { background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }
    .token-card { background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 6px; padding: 6px 12px; margin: 6px 0; display: flex; gap: 12px; font-size: 0.8rem; font-weight: 600; flex-wrap: wrap; }
    .source-box { background: rgba(128, 128, 128, 0.05); border-left: 3px solid #3b82f6; border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 6px 0 10px 0; font-size: 0.85rem; line-height: 1.4; font-family: monospace; white-space: pre-wrap; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.72rem; font-weight: 600; background: rgba(59, 130, 246, 0.15); color: #3b82f6; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------- STATE INITIALIZATION ----------------
for key, default in [("chat_history", []), ("vector_store", None), ("doc_metadata", {}), ("files_hash", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.title("📚 DocuQuery AI")
    
    # 1. Engine Selection
    llm_choice = st.radio("🧠 LLM Engine", ["Groq (Ultra-Fast LPU)", "Local Ollama", "Google Gemini"], index=0)
    provider = "groq" if "Groq" in llm_choice else ("ollama" if "Ollama" in llm_choice else "gemini")
    
    api_key, base_url, model_name = None, "http://localhost:11434", ""
    env_groq = os.getenv("GROQ_API_KEY", "").strip()
    env_gemini = os.getenv("GOOGLE_API_KEY", "").strip()

    if provider == "groq":
        user_groq = st.text_input(
            "⚡ Groq API Key",
            value="",
            placeholder="✅ Configured via Secrets" if env_groq else "Enter Groq API Key (gsk_...)",
            type="password",
            help="Your API key stays safe on the server and is never exposed in browser HTML."
        )
        api_key = user_groq.strip() if user_groq.strip() else env_groq
        model_name = st.selectbox("Model", [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "groq/compound-mini",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ])
    elif provider == "gemini":
        user_gemini = st.text_input(
            "🔑 Gemini API Key",
            value="",
            placeholder="✅ Configured via Secrets" if env_gemini else "Enter Gemini API Key (AIzaSy...)",
            type="password",
            help="Your API key stays safe on the server and is never exposed in browser HTML."
        )
        api_key = user_gemini.strip() if user_gemini.strip() else env_gemini
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"])
    else:
        base_url = st.text_input("🌐 Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        model_name = st.selectbox("Model", ["llama3", "llama3.2", "mistral", "deepseek-r1", "phi3"])

    st.markdown("---")
    
    # 2. Document Upload
    uploaded_files = st.file_uploader("Upload Documents (PDF, Word, TXT, Images)", type=["pdf", "docx", "txt", "md", "png", "jpg"], accept_multiple_files=True)
    enable_crag = st.toggle("⚡ Enable LangGraph CRAG", value=True)
    enable_web = st.toggle("🌐 Enable Web Search Comparison", value=True, help="Searches the live web to compare or enrich context when documents lack details.")
    
    with st.expander("⚙️ Advanced Parameters"):
        temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("Top-k Passages", 2, 12, 5, 1)
        chunk_size = st.slider("Chunk Size", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200, 50)
        embed_provider = st.selectbox("Embedding", ["huggingface", "gemini", "ollama"], index=0)

    # 3. Document Ingestion Pipeline
    if uploaded_files:
        cur_hash = f"{'_'.join(f.name + str(f.size) for f in uploaded_files)}_{embed_provider}"
        if st.session_state.files_hash != cur_hash:
            with st.spinner(f"Indexing {len(uploaded_files)} document(s)..."):
                try:
                    all_chunks, total_pages, books_info = [], 0, []
                    for f in uploaded_files:
                        raw_docs = load_document_from_bytes(f.read(), f.name)
                        pages = max((d.metadata.get("page", 1) for d in raw_docs), default=1) if raw_docs else 1
                        total_pages += pages
                        chunks = split_documents_into_chunks(raw_docs, chunk_size, chunk_overlap)
                        all_chunks.extend(chunks)
                        books_info.append({"name": f.name, "pages": pages, "chunks": len(chunks)})

                    embeddings = get_embedding_function(embed_provider, api_key if embed_provider == "gemini" else None)
                    st.session_state.vector_store = build_vector_store(all_chunks, embeddings, persist_directory="./chroma_db")
                    st.session_state.files_hash = cur_hash
                    st.session_state.doc_metadata = {"total_books": len(uploaded_files), "total_pages": total_pages, "total_chunks": len(all_chunks), "books": books_info}
                    st.session_state.chat_history = []
                    st.success(f"Indexed {len(uploaded_files)} Doc(s) • {total_pages} Pages • {len(all_chunks)} Chunks!")
                except Exception as e:
                    st.error(f"Ingestion Error: {e}")

    # 4. Library Metrics & Reset
    if st.session_state.doc_metadata:
        m = st.session_state.doc_metadata
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Docs", m['total_books'])
        c2.metric("Pages", m['total_pages'])
        c3.metric("Chunks", m['total_chunks'])
        if st.button("🗑️ Reset Library & Chat", use_container_width=True):
            clear_persisted_vector_store("./chroma_db")
            st.session_state.update({"vector_store": None, "doc_metadata": {}, "chat_history": [], "files_hash": None})
            st.rerun()

# ---------------- MAIN UI & CHAT ----------------
st.markdown('<div class="main-header">📚 DocuQuery AI</div>', unsafe_allow_html=True)
st.caption(f"⚡ {'LangGraph CRAG Active' if enable_crag else 'Standard RAG'} • **{provider.upper()}** ({model_name}) • Grounded Multi-Document Assistant.")

if not st.session_state.vector_store:
    st.info("👈 **Get Started**: Upload PDF, Word, TXT, or Image files in the sidebar to build your Knowledge Library.")
else:
    meta = st.session_state.doc_metadata
    st.markdown(f'<div class="doc-banner"><span><b>Active Library:</b> {meta["total_books"]} Document(s)</span><span><span class="badge">{meta["total_pages"]} Pages</span><span class="badge">{meta["total_chunks"]} Chunks</span><span class="badge">{provider.upper()}: {model_name}</span></span></div>', unsafe_allow_html=True)

    # Automated RAG Triad Benchmark Hub
    with st.expander("📊 RAG Triad Evaluation & Benchmark Hub", expanded=False):
        if st.button("🧪 Run Automated RAG Triad Benchmark", use_container_width=True):
            with st.spinner("Auditing RAG pipeline against synthetic test cases..."):
                try:
                    eval_llm = get_llm(provider, model_name, 0.0, api_key, base_url)
                    retriever = get_retriever(st.session_state.vector_store, k=top_k)
                    test_cases = generate_synthetic_benchmark_dataset(retriever.invoke("summary and core rules"), eval_llm, num_questions=3)
                    
                    scores_list = []
                    for t in test_cases:
                        r_docs = retriever.invoke(t["question"])
                        ans = eval_llm.invoke(f"Context:\n{format_docs_with_metadata(r_docs)}\n\nQuestion: {t['question']}").content
                        scores_list.append((t["question"], evaluate_rag_triad(t["question"], ans, r_docs, eval_llm)))
                    
                    avg_score = round(sum(s[1]["overall_score"] for s in scores_list) / len(scores_list), 1)
                    st.metric("🏆 Overall RAG Quality Score", f"{avg_score}%")
                    st.dataframe([{"Question": s[0], "Context Rel.": f"{s[1]['context_relevance']['score']}%", "Faithfulness": f"{s[1]['faithfulness']['score']}%", "Answer Rel.": f"{s[1]['answer_relevance']['score']}%", "Composite": f"{s[1]['overall_score']}%", "Status": s[1]['status']} for s in scores_list], use_container_width=True)
                except Exception as e:
                    st.error(f"Benchmark Error: {e}")

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if ci := msg.get("crag_info"):
                cost = f"${ci.get('estimated_cost_usd', 0.0):.5f}" if ci.get('estimated_cost_usd', 0.0) > 0 else "Free ($0.00)"
                savings = f" | 📉 {ci.get('token_savings_pct', 0)}% Saved" if ci.get('token_savings_pct', 0) > 0 else ""
                web_tag = " | 🌐 Web Augmented" if ci.get('web_search_used') else ""
                st.markdown(f'<div class="token-card"><span>🪙 {ci.get("total_tokens", 0)} Toks ({ci.get("input_tokens", 0)} in / {ci.get("output_tokens", 0)} out)</span><span>💰 {cost}</span><span>🛡️ {ci.get("groundedness_score", 100)}% Grounded{savings}{web_tag}</span></div>', unsafe_allow_html=True)
            if srcs := msg.get("sources"):
                with st.expander(f"📖 View Referenced Sources ({len(srcs)} chunks)", expanded=False):
                    for idx, s in enumerate(srcs):
                        st.markdown(f"**Source #{idx+1} — `{s['source']}` | Page {s['page']}**\n<div class=\"source-box\">{s['content']}</div>", unsafe_allow_html=True)

    # Handle User Query
    user_query = st.chat_input("Ask any question across your uploaded documents...")
    if user_query:
        if provider in ["gemini", "groq"] and not api_key:
            st.error(f"⚠️ Please enter your {provider.title()} API Key in the sidebar.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                retriever = get_retriever(st.session_state.vector_store, k=top_k)
                hist = format_chat_history(st.session_state.chat_history[:-1])

                if enable_crag:
                    with st.spinner("⚡ Executing LangGraph CRAG & Token Pruning..."):
                        stream, source_docs, crag_stats = stream_langgraph_crag_pipeline(
                            user_query, retriever, hist, provider, model_name, temp, api_key, base_url, enable_web_search=enable_web
                        )
                    answer_text = st.write_stream(stream)
                    if crag_stats:
                        cost = f"${crag_stats.get('estimated_cost_usd', 0.0):.5f}" if crag_stats.get('estimated_cost_usd', 0.0) > 0 else "Free ($0.00)"
                        savings = f" | 📉 {crag_stats.get('token_savings_pct', 0)}% Saved" if crag_stats.get('token_savings_pct', 0) > 0 else ""
                        web_tag = " | 🌐 Web Augmented" if crag_stats.get('web_search_used') else ""
                        st.markdown(f'<div class="token-card"><span>🪙 {crag_stats.get("total_tokens", 0)} Toks ({crag_stats.get("input_tokens", 0)} in / {crag_stats.get("output_tokens", 0)} out)</span><span>💰 {cost}</span><span>🛡️ {crag_stats.get("groundedness_score", 100)}% Grounded{savings}{web_tag}</span></div>', unsafe_allow_html=True)
                else:
                    llm = get_llm(provider, model_name, temp, api_key, base_url)
                    with st.spinner("Generating grounded answer..."):
                        res = create_rag_chain(retriever, llm).invoke({"question": user_query, "chat_history": hist})
                    answer_text, source_docs, crag_stats = res["answer"], res["source_documents"], None
                    st.markdown(answer_text)

                sources_data = [{"page": d.metadata.get("page", 1), "source": d.metadata.get("source", "Doc"), "content": d.page_content} for d in source_docs]
                if sources_data:
                    with st.expander(f"📖 View Referenced Sources ({len(sources_data)} chunks)", expanded=False):
                        for idx, s in enumerate(sources_data):
                            st.markdown(f"**Source #{idx+1} — `{s['source']}` | Page {s['page']}**\n<div class=\"source-box\">{s['content']}</div>", unsafe_allow_html=True)

                st.session_state.chat_history.append({"role": "assistant", "content": answer_text, "sources": sources_data, "crag_info": crag_stats})

    # Download Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        export_text = f"# DocuQuery AI Conversation History\n\n" + "\n\n".join(f"### {'🧑 User' if m['role'] == 'user' else '🤖 Assistant'}\n{m['content']}" for m in st.session_state.chat_history)
        st.download_button("📥 Download Conversation History (Markdown)", data=export_text, file_name="conversation_history.md", mime="text/markdown", use_container_width=True)
