import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from rag_engine.document_loader import load_document_from_bytes
from rag_engine.text_splitter import split_documents_into_chunks
from rag_engine.vector_store import (
    get_embedding_function,
    build_vector_store,
    get_retriever,
    load_persisted_vector_store,
    clear_persisted_vector_store,
)
from rag_engine.chain import create_rag_chain, format_chat_history, get_llm
from rag_engine.crag import run_crag_pipeline, stream_crag_pipeline
from rag_engine.vision_comparator import analyze_and_compare_image_with_book
from rag_engine.market_data import (
    fetch_live_ticker_data,
    create_interactive_candlestick_chart,
    render_candlestick_image_bytes,
    calculate_market_summary,
)


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="DocuQuery AI - Multi-Document RAG & Live Market Vision Comparator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-Adaptive CSS Styling (Dark & Light Mode Friendly)
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 0.98rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }
    
    /* Theme-adaptive active document card */
    .doc-banner {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    /* Theme-adaptive Citation box */
    .source-box {
        background: rgba(128, 128, 128, 0.07);
        border-left: 3.5px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        line-height: 1.5;
        border-top: 1px solid rgba(128, 128, 128, 0.12);
        border-right: 1px solid rgba(128, 128, 128, 0.12);
        border-bottom: 1px solid rgba(128, 128, 128, 0.12);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* Comparison report container */
    .report-box {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-right: 6px;
    }
    .badge-crag {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
        margin-right: 6px;
    }
    .badge-ollama {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.3);
        margin-right: 6px;
    }
    .feature-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE INITIALIZATION ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = {}

if "processed_files_hash" not in st.session_state:
    st.session_state.processed_files_hash = None

if "last_comparison_report" not in st.session_state:
    st.session_state.last_comparison_report = None

if "live_df" not in st.session_state:
    st.session_state.live_df = None

if "live_ticker" not in st.session_state:
    st.session_state.live_ticker = ""


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/document.png", width=52)
    st.title("Settings & Library")

    # LLM Provider Selection
    llm_provider = st.radio(
        "🧠 LLM Engine",
        options=["Local Ollama", "Google Gemini"],
        index=0,
        help="Choose between 100% offline local Ollama models or cloud Google Gemini.",
    )
    provider_key = "ollama" if llm_provider == "Local Ollama" else "gemini"

    api_key = None
    ollama_base_url = "http://localhost:11434"
    model_name = ""

    if provider_key == "ollama":
        ollama_base_url = st.text_input(
            "🌐 Ollama Server URL",
            value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            help="Default is http://localhost:11434. Ensure Ollama is running (`ollama serve`).",
        )
        model_choice = st.selectbox(
            "Local Model",
            options=["llama3", "llama3.2", "mistral", "deepseek-r1", "phi3", "qwen2.5", "gemma2", "Custom"],
            index=0,
            help="For Visual Chart Comparison with Ollama, ensure you pull a vision model like `llama3.2-vision` or `llava`.",
        )
        if model_choice == "Custom":
            model_name = st.text_input("Custom Ollama Model Name", value="llama3")
        else:
            model_name = model_choice
    else:
        env_api_key = os.getenv("GOOGLE_API_KEY", "")
        api_key_input = st.text_input(
            "🔑 Gemini API Key",
            value=env_api_key,
            type="password",
            help="Get a free key from https://aistudio.google.com/ or set GOOGLE_API_KEY in .env",
        )
        api_key = api_key_input.strip() if api_key_input else env_api_key
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0,
        )

    st.markdown("---")

    # Multi-Document Knowledge Library Uploader
    st.subheader("📚 Knowledge Library (Multi-Books)")
    uploaded_files = st.file_uploader(
        "Upload Reference Books / Guides (PDF, Word, TXT)",
        type=["pdf", "docx", "txt", "md", "png", "jpg"],
        accept_multiple_files=True,
        help="Upload one or multiple books (e.g. Candlestick Guide + Volume Analysis + Risk Management).",
    )

    # RAG Mode Selection
    enable_crag = st.toggle(
        "⚡ Enable Corrective RAG (CRAG)",
        value=True,
        help="Grades chunks for relevance, filters out noise, and eliminates hallucinations.",
    )

    # Advanced Settings
    with st.expander("⚙️ Advanced Parameters", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower values yield more strictly grounded answers.",
        )
        top_k = st.slider(
            "Retrieved Chunks ($k$)",
            min_value=2,
            max_value=12,
            value=5,
            step=1,
            help="Number of document passages retrieved from across all uploaded books.",
        )
        chunk_size = st.slider(
            "Chunk Size (chars)",
            min_value=300,
            max_value=2500,
            value=1000,
            step=100,
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (chars)",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
        )
        if provider_key == "ollama":
            embedding_provider = st.selectbox(
                "Embedding Provider",
                options=["ollama", "huggingface"],
                index=0,
                help="Ollama embeddings (`nomic-embed-text`) or local HuggingFace (`all-MiniLM-L6-v2`).",
            )
        else:
            embedding_provider = st.selectbox(
                "Embedding Provider",
                options=["gemini", "huggingface"],
                index=0,
                help="Gemini embeddings or local HuggingFace.",
            )

    st.markdown("---")

    # Multi-Document Ingestion Pipeline
    if uploaded_files:
        current_files_hash = "_".join([f"{f.name}_{f.size}" for f in uploaded_files]) + f"_{embedding_provider}_{provider_key}"
        
        # Process library if new files or parameters changed
        if st.session_state.processed_files_hash != current_files_hash:
            if provider_key == "gemini" and not api_key and embedding_provider == "gemini":
                st.error("⚠️ Please enter a Google API Key above to process documents with Gemini embeddings.")
            else:
                with st.spinner(f"Processing & Indexing {len(uploaded_files)} book(s) in Knowledge Library..."):
                    try:
                        all_chunks = []
                        total_library_pages = 0
                        books_info = []

                        for uploaded_file in uploaded_files:
                            file_bytes = uploaded_file.read()
                            raw_docs = load_document_from_bytes(file_bytes, uploaded_file.name)
                            doc_pages = max([d.metadata.get("page", 1) for d in raw_docs]) if raw_docs else 0
                            total_library_pages += doc_pages
                            
                            chunks = split_documents_into_chunks(
                                raw_docs,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                            )
                            all_chunks.extend(chunks)
                            books_info.append({
                                "name": uploaded_file.name,
                                "pages": doc_pages,
                                "chunks": len(chunks),
                                "size_kb": round(uploaded_file.size / 1024, 2),
                            })

                        # Build Vector Store with Disk Persistence
                        embeddings = get_embedding_function(
                            provider=embedding_provider,
                            api_key=api_key if embedding_provider == "gemini" else None,
                        )
                        vector_store = build_vector_store(
                            chunks=all_chunks,
                            embedding_function=embeddings,
                            persist_directory="./chroma_db",
                        )

                        # Save state
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_files_hash = current_files_hash
                        st.session_state.doc_metadata = {
                            "total_books": len(uploaded_files),
                            "total_pages": total_library_pages,
                            "total_chunks": len(all_chunks),
                            "books": books_info,
                        }
                        st.session_state.chat_history = []
                        st.session_state.last_comparison_report = None
                        st.success(f"Indexed {len(uploaded_files)} Book(s) • {total_library_pages} Total Pages • {len(all_chunks)} Chunks!")
                    except Exception as e:
                        st.error(f"Error processing Knowledge Library: {str(e)}")

    # Sidebar Knowledge Library Summary
    if st.session_state.doc_metadata:
        meta = st.session_state.doc_metadata
        st.subheader("📊 Library Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Books", meta['total_books'])
        with col_m2:
            st.metric("Pages", meta['total_pages'])
        with col_m3:
            st.metric("Chunks", meta['total_chunks'])
        
        with st.expander("📚 Indexed Books List", expanded=False):
            for b in meta.get("books", []):
                st.markdown(f"- **`{b['name']}`**: {b['pages']} pages ({b['chunks']} chunks)")

        if st.button("🗑️ Reset Library & Chat", use_container_width=True):
            clear_persisted_vector_store("./chroma_db")
            st.session_state.vector_store = None
            st.session_state.doc_metadata = {}
            st.session_state.chat_history = []
            st.session_state.processed_files_hash = None
            st.session_state.last_comparison_report = None
            st.rerun()


# ---------------- MAIN UI & TABS ----------------
st.markdown('<div class="main-header">📈 DocuQuery AI</div>', unsafe_allow_html=True)
mode_desc = "⚡ Corrective RAG (CRAG) Active" if enable_crag else "Standard RAG Active"
st.markdown(f'<div class="sub-header">{mode_desc} • <b>{llm_provider}</b> ({model_name}) • Multi-Book Library Q&A & Live Candlestick Pattern Comparator.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["💬 Knowledge Library Chat (RAG / CRAG)", "📈 Live Market Data & Pattern Comparator"])


# =========================================================================
# TAB 1: CONVERSATIONAL MULTI-BOOK LIBRARY Q&A (RAG / CRAG)
# =========================================================================
with tab1:
    if st.session_state.vector_store is None:
        st.info("👈 **Get Started**: Upload one or multiple reference books (PDF, Word, or Text) in the sidebar to build your Knowledge Library.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>📚 Multi-Document Library</h4>
                <p style="font-size:0.9rem; opacity:0.85;">Upload multiple books at once (e.g. Candlestick Guide + Volume Analysis + Risk Management). Chunks track book source and page numbers.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🦙 Local Ollama & Gemini</h4>
                <p style="font-size:0.9rem; opacity:0.85;">Run 100% locally and privately with <b>Ollama</b> (Llama 3, DeepSeek, Mistral) or in the cloud with <b>Google Gemini</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>🛡️ Corrective RAG (CRAG)</h4>
                <p style="font-size:0.9rem; opacity:0.85;">An internal evaluator grades retrieved chunks across all books for relevance to eliminate noise and prevent hallucinations.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        meta = st.session_state.doc_metadata
        crag_tag = '<span class="badge-crag">CRAG: Active</span>' if enable_crag else '<span class="badge">Standard RAG</span>'
        prov_tag = f'<span class="badge-ollama">Ollama: {model_name}</span>' if provider_key == "ollama" else f'<span class="badge">Gemini: {model_name}</span>'
        
        st.markdown(
            f"""
            <div class="doc-banner">
                <div>
                    <b>Active Library:</b> <code>{meta['total_books']} Book(s) Loaded</code>
                </div>
                <div>
                    <span class="badge">{meta['total_pages']} Total Pages</span>
                    <span class="badge">{meta['total_chunks']} Chunks</span>
                    {prov_tag}
                    {crag_tag}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested Prompts
        if len(st.session_state.chat_history) == 0:
            st.markdown("##### 💡 Suggested Questions across Library:")
            quick_cols = st.columns(3)
            with quick_cols[0]:
                if st.button("📌 Synthesize candlestick patterns & volume", use_container_width=True):
                    st.session_state.quick_query = "Synthesize the main candlestick patterns and how volume confirms each reversal across the uploaded books."
            with quick_cols[1]:
                if st.button("🔑 Risk management & stop-loss rules", use_container_width=True):
                    st.session_state.quick_query = "What are the recommended stop-loss, risk-to-reward, and position sizing rules described in the library?"
            with quick_cols[2]:
                if st.button("📋 Reversal vs Continuation criteria", use_container_width=True):
                    st.session_state.quick_query = "Compare the validation criteria for Bullish Reversal patterns vs Trend Continuation patterns from the text."

        # Render Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("crag_info"):
                    ci = msg["crag_info"]
                    st.caption(f"🔍 **CRAG Evaluation:** Evaluated {ci.get('initial_retrieved', 0)} chunks → Kept {ci.get('relevant_filtered', 0)} relevant" + (f" | Query Rewritten: *{ci.get('rewritten_query')}*" if ci.get('query_rewritten') else ""))
                if msg.get("sources"):
                    with st.expander(f"📖 View Referenced Sources ({len(msg['sources'])} chunks across books)", expanded=False):
                        for idx, src in enumerate(msg["sources"]):
                            page_num = src.get("page", "Unknown")
                            chunk_id = src.get("chunk_id", idx + 1)
                            source_file = src.get("source", "Document")
                            snippet = src.get("content", "")
                            st.markdown(f"**Source #{idx+1} — `{source_file}` | Page {page_num} (Chunk {chunk_id})**")
                            st.markdown(f'<div class="source-box">{snippet}</div>', unsafe_allow_html=True)

        # Handle User Query
        user_query = st.chat_input("Ask any question across your uploaded books...")

        if "quick_query" in st.session_state and st.session_state.quick_query:
            user_query = st.session_state.quick_query
            st.session_state.quick_query = None

        if user_query:
            if provider_key == "gemini" and not api_key:
                st.error("⚠️ Please enter a Google API Key in the sidebar.")
            else:
                # 1. User message
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                # 2. Assistant generation with streaming
                with st.chat_message("assistant"):
                    retriever = get_retriever(st.session_state.vector_store, k=top_k)
                    langchain_history = format_chat_history(st.session_state.chat_history[:-1])

                    if enable_crag:
                        with st.spinner("Grading chunks with CRAG across library..."):
                            token_stream, source_docs, crag_stats = stream_crag_pipeline(
                                question=user_query,
                                retriever=retriever,
                                chat_history=langchain_history,
                                provider=provider_key,
                                model_name=model_name,
                                temperature=temperature,
                                api_key=api_key,
                                base_url=ollama_base_url,
                            )
                        answer_text = st.write_stream(token_stream)
                        if crag_stats:
                            st.caption(f"🔍 **CRAG Evaluation:** Evaluated {crag_stats['initial_retrieved']} chunks → Kept {crag_stats['relevant_filtered']} relevant" + (f" | Query Rewritten: *{crag_stats['rewritten_query']}*" if crag_stats['query_rewritten'] else ""))
                    else:
                        llm_instance = get_llm(
                            provider=provider_key,
                            model_name=model_name,
                            temperature=temperature,
                            api_key=api_key,
                            base_url=ollama_base_url,
                        )
                        rag_chain = create_rag_chain(
                            retriever=retriever,
                            llm=llm_instance,
                        )
                        with st.spinner("Generating grounded answer from library..."):
                            result = rag_chain.invoke({
                                "question": user_query,
                                "chat_history": langchain_history,
                            })
                        answer_text = result["answer"]
                        source_docs = result["source_documents"]
                        crag_stats = None
                        st.markdown(answer_text)

                    # Extract source snippets
                    sources_data = []
                    for doc in source_docs:
                        sources_data.append({
                            "page": doc.metadata.get("page", "Unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            "source": doc.metadata.get("source", "Document"),
                            "content": doc.page_content,
                        })

                    # Render Sources Accordion
                    if sources_data:
                        with st.expander(f"📖 View Referenced Sources ({len(sources_data)} chunks across books)", expanded=False):
                            for idx, src in enumerate(sources_data):
                                page_num = src.get("page", "Unknown")
                                chunk_id = src.get("chunk_id", idx + 1)
                                source_file = src.get("source", "Document")
                                snippet = src.get("content", "")
                                st.markdown(f"**Source #{idx+1} — `{source_file}` | Page {page_num} (Chunk {chunk_id})**")
                                st.markdown(f'<div class="source-box">{snippet}</div>', unsafe_allow_html=True)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_data,
                        "crag_info": crag_stats,
                    })

        # Download Chat History
        if len(st.session_state.chat_history) > 0:
            st.markdown("---")
            chat_export_text = "# DocuQuery AI - Multi-Book Conversation History\n\n"
            if st.session_state.doc_metadata:
                chat_export_text += f"**Library:** {st.session_state.doc_metadata.get('total_books')} books ({st.session_state.doc_metadata.get('total_pages')} total pages)\n\n---\n\n"
            for msg in st.session_state.chat_history:
                speaker = "🧑 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
                chat_export_text += f"### {speaker}\n{msg['content']}\n\n"
            
            st.download_button(
                label="📥 Download Library Chat (Markdown)",
                data=chat_export_text,
                file_name="library_conversation_history.md",
                mime="text/markdown",
                use_container_width=True,
            )


# =========================================================================
# TAB 2: LIVE MARKET DATA & VISUAL PATTERN COMPARATOR
# =========================================================================
with tab2:
    st.subheader("📈 Live Market Data Feed & Pattern Comparator")
    st.markdown("Fetch live stock, crypto, or index candlestick charts in real time, or upload custom screenshots. The system visually analyzes price action and cross-examines pattern criteria against your uploaded reference books.")

    if st.session_state.vector_store is None:
        st.warning("⚠️ **Please upload at least one Reference Book / Technical Guide in the sidebar first!** The comparator needs your book library to match patterns against.")
    else:
        source_mode = st.radio(
            "Select Chart Source",
            options=["⚡ Live Ticker Search (Yahoo Finance)", "📷 Upload Custom Screenshot"],
            horizontal=True,
        )

        chart_image_bytes = None
        chart_title = ""
        market_stats_text = ""

        # MODE A: LIVE TICKER SEARCH
        if source_mode == "⚡ Live Ticker Search (Yahoo Finance)":
            col_t1, col_t2, col_t3, col_t4 = st.columns([2, 1, 1, 1])
            with col_t1:
                ticker_input = st.text_input("Stock / Crypto / Index Ticker", value="AAPL", placeholder="e.g. AAPL, NVDA, TSLA, BTC-USD, RELIANCE.NS, ^NSEI")
            with col_t2:
                time_period = st.selectbox("Period", options=["1mo", "3mo", "6mo", "1y", "2y"], index=1)
            with col_t3:
                time_interval = st.selectbox("Interval", options=["1d", "1wk", "1h", "15m", "5m"], index=0)
            with col_t4:
                st.write("")
                st.write("")
                fetch_btn = st.button("📥 Fetch Live Chart", use_container_width=True)

            if fetch_btn and ticker_input:
                with st.spinner(f"Fetching live OHLCV data for {ticker_input.upper()}..."):
                    df, err = fetch_live_ticker_data(ticker_input, period=time_period, interval=time_interval)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.live_df = df
                        st.session_state.live_ticker = ticker_input.upper()

            # Display Live Chart & Summary
            if st.session_state.live_df is not None:
                df = st.session_state.live_df
                cur_ticker = st.session_state.live_ticker
                stats = calculate_market_summary(df, cur_ticker)

                # Metrics banner
                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                with c_m1:
                    change_sign = "+" if stats['change'] >= 0 else ""
                    st.metric("Latest Price", f"${stats['last_price']}", f"{change_sign}{stats['change_pct']}%")
                with c_m2:
                    st.metric("Period High", f"${stats['high']}")
                with c_m3:
                    st.metric("Period Low", f"${stats['low']}")
                with c_m4:
                    st.metric("Volume", f"{stats['last_volume']:,}")

                # Interactive Plotly Chart
                plotly_fig = create_interactive_candlestick_chart(df, cur_ticker)
                st.plotly_chart(plotly_fig, use_container_width=True)

                # Render high-res image buffer for Vision Model
                chart_image_bytes = render_candlestick_image_bytes(df, cur_ticker)
                chart_title = f"Live Chart: {cur_ticker} ({time_interval}, {time_period})"
                market_stats_text = f"Live Market Context: Ticker={cur_ticker}, Last Close=${stats['last_price']}, Change={stats['change_pct']}%, High=${stats['high']}, Low=${stats['low']}, Volume={stats['last_volume']}."
            else:
                st.info("💡 Enter any stock/crypto ticker symbol above (e.g. `AAPL`, `NVDA`, `BTC-USD`) and click **'📥 Fetch Live Chart'** to view live price action.")

        # MODE B: CUSTOM SCREENSHOT UPLOAD
        else:
            custom_chart_file = st.file_uploader(
                "Upload Candlestick Chart, Trading Screenshot, or Diagram",
                type=["png", "jpg", "jpeg", "webp"],
                key="custom_chart_uploader",
            )
            if custom_chart_file is not None:
                st.image(custom_chart_file, caption=f"Uploaded Chart: {custom_chart_file.name}", use_container_width=True)
                chart_image_bytes = custom_chart_file.getvalue()
                chart_title = custom_chart_file.name

        # Comparative Analysis Section
        st.markdown("---")
        st.markdown("#### 🎯 Run Pattern Verification against Reference Books")
        
        custom_analysis_prompt = st.text_area(
            "Specific Focus / Questions (Optional)",
            placeholder="e.g. Look for bottom reversal patterns, check volume confirmation against Volume Analysis book, and identify stop-loss rules from Risk Management manual.",
            height=80,
        )

        compare_live_btn = st.button("🚀 Compare Chart with Knowledge Library & Predict Pattern", type="primary", use_container_width=True)

        if compare_live_btn:
            if chart_image_bytes is None:
                st.warning("⚠️ Please fetch a live chart or upload an image first.")
            elif provider_key == "gemini" and not api_key:
                st.error("⚠️ Please enter a Google API Key in the sidebar.")
            else:
                combined_instructions = f"{market_stats_text}\n{custom_analysis_prompt}".strip()
                with st.spinner("🔍 Inspecting visual chart candlesticks, retrieving book rules across library, and synthesizing comparison..."):
                    try:
                        report_data = analyze_and_compare_image_with_book(
                            image_bytes=chart_image_bytes,
                            image_filename=chart_title or "live_chart.png",
                            vector_store=st.session_state.vector_store,
                            provider=provider_key,
                            model_name=model_name,
                            api_key=api_key,
                            base_url=ollama_base_url,
                            user_instructions=combined_instructions,
                            top_k=top_k,
                        )
                        st.session_state.last_comparison_report = report_data
                        st.success("✅ Multi-Book Pattern Comparison Complete!")
                    except Exception as e:
                        st.error(f"Error during comparison: {str(e)}")

        # Display Last Comparison Report
        if st.session_state.last_comparison_report:
            rep = st.session_state.last_comparison_report
            st.markdown("---")
            st.markdown("### 📊 Comparative Analysis Report: Book Theory vs. Real Chart")
            
            with st.expander("🔍 View Raw Visual Inspection Notes", expanded=False):
                st.markdown(rep.get("visual_analysis", ""))

            # Main report
            st.markdown(f'<div class="report-box">{rep.get("comparison_report", "")}</div>', unsafe_allow_html=True)

            # Referenced Book Sources across Library
            if rep.get("source_documents"):
                with st.expander(f"📖 View Matching Reference Book Pages ({len(rep['source_documents'])} chunks across library)", expanded=False):
                    for idx, doc in enumerate(rep["source_documents"]):
                        page_num = doc.metadata.get("page", "Unknown")
                        chunk_id = doc.metadata.get("chunk_id", idx + 1)
                        source_file = doc.metadata.get("source", "Document")
                        snippet = doc.page_content
                        st.markdown(f"**Reference #{idx+1} — `{source_file}` | Page {page_num} (Chunk {chunk_id})**")
                        st.markdown(f'<div class="source-box">{snippet}</div>', unsafe_allow_html=True)

            # Export Report
            st.download_button(
                label="📥 Download Comparison Report (Markdown)",
                data=f"# Visual Pattern Comparison Report\n\n**Chart:** {chart_title}\n\n---\n\n{rep.get('comparison_report', '')}",
                file_name="pattern_comparison_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
