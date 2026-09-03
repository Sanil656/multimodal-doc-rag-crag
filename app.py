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
from rag_engine.crag_graph import stream_langgraph_crag_pipeline


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="DocuQuery AI - Multi-Document RAG & Corrective RAG (CRAG) Assistant",
    page_icon="📚",
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
    .badge-groq {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
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


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/document.png", width=52)
    st.title("Settings & Library")

    # LLM Provider Selection
    llm_provider = st.radio(
        "🧠 LLM Engine",
        options=["Local Ollama", "Google Gemini", "Groq (Ultra-Fast LPU)"],
        index=0,
        help="Choose between 100% offline local Ollama, cloud Google Gemini, or ultra-fast Groq LPU inference.",
    )
    if llm_provider == "Local Ollama":
        provider_key = "ollama"
    elif llm_provider == "Google Gemini":
        provider_key = "gemini"
    else:
        provider_key = "groq"

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
            help="Any pulled model from Ollama.",
        )
        if model_choice == "Custom":
            model_name = st.text_input("Custom Ollama Model Name", value="llama3")
        else:
            model_name = model_choice

    elif provider_key == "gemini":
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

    else:  # Groq Provider
        env_groq_key = os.getenv("GROQ_API_KEY", "")
        groq_key_input = st.text_input(
            "⚡ Groq API Key",
            value=env_groq_key,
            type="password",
            help="Get a free ultra-fast key from https://console.groq.com/ or set GROQ_API_KEY in .env",
        )
        api_key = groq_key_input.strip() if groq_key_input else env_groq_key
        groq_model_choice = st.selectbox(
            "Groq Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "deepseek-r1-distill-llama-70b",
                "mixtral-8x7b-32768",
                "Custom",
            ],
            index=0,
            help="Flagship llama-3.3-70b, deep reasoning DeepSeek-R1, or ultra-fast 8B.",
        )
        if groq_model_choice == "Custom":
            model_name = st.text_input("Custom Groq Model Name", value="llama-3.3-70b-versatile")
        else:
            model_name = groq_model_choice

    st.markdown("---")

    # Multi-Document Knowledge Library Uploader
    st.subheader("📚 Knowledge Library (Multi-Books)")
    uploaded_files = st.file_uploader(
        "Upload Documents / Books (PDF, Word, TXT, Images)",
        type=["pdf", "docx", "txt", "md", "png", "jpg"],
        accept_multiple_files=True,
        help="Upload one or multiple documents at once (1 to 1000+ pages).",
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
        elif provider_key == "gemini":
            embedding_provider = st.selectbox(
                "Embedding Provider",
                options=["gemini", "huggingface"],
                index=0,
                help="Gemini embeddings or local HuggingFace.",
            )
        else:
            embedding_provider = st.selectbox(
                "Embedding Provider",
                options=["huggingface", "gemini", "ollama"],
                index=0,
                help="Local HuggingFace embeddings (`all-MiniLM-L6-v2`), Gemini, or Ollama.",
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
                with st.spinner(f"Processing & Indexing {len(uploaded_files)} document(s) in Knowledge Library..."):
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
                        st.success(f"Indexed {len(uploaded_files)} Document(s) • {total_library_pages} Total Pages • {len(all_chunks)} Chunks!")
                    except Exception as e:
                        st.error(f"Error processing Knowledge Library: {str(e)}")

    # Sidebar Knowledge Library Summary
    if st.session_state.doc_metadata:
        meta = st.session_state.doc_metadata
        st.subheader("📊 Library Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Documents", meta['total_books'])
        with col_m2:
            st.metric("Pages", meta['total_pages'])
        with col_m3:
            st.metric("Chunks", meta['total_chunks'])
        
        with st.expander("📚 Indexed Documents List", expanded=False):
            for b in meta.get("books", []):
                st.markdown(f"- **`{b['name']}`**: {b['pages']} pages ({b['chunks']} chunks)")

        if st.button("🗑️ Reset Library & Chat", use_container_width=True):
            clear_persisted_vector_store("./chroma_db")
            st.session_state.vector_store = None
            st.session_state.doc_metadata = {}
            st.session_state.chat_history = []
            st.session_state.processed_files_hash = None
            st.rerun()


# ---------------- MAIN UI ----------------
st.markdown('<div class="main-header">📚 DocuQuery AI</div>', unsafe_allow_html=True)
mode_desc = "⚡ Corrective RAG (CRAG) Active" if enable_crag else "Standard RAG Active"
st.markdown(f'<div class="sub-header">{mode_desc} • <b>{llm_provider}</b> ({model_name}) • Enterprise Multi-Document Q&A with Grounded Page Citations.</div>', unsafe_allow_html=True)

if st.session_state.vector_store is None:
    st.info("👈 **Get Started**: Upload one or multiple documents (PDF, Word, TXT, or Images) in the sidebar to build your Knowledge Library.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📚 Multi-Document Ingestion</h4>
            <p style="font-size:0.9rem; opacity:0.85;">Upload multi-page documents (1 to 1000+ pages) in PDF, Word, TXT, or image OCR format. Chunks preserve book titles and page metadata.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚡ Groq, Ollama & Gemini</h4>
            <p style="font-size:0.9rem; opacity:0.85;">Switch between ultra-fast <b>Groq LPU</b> (500+ tokens/sec), 100% private offline <b>Local Ollama</b>, or <b>Google Gemini</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🛡️ Corrective RAG (CRAG)</h4>
            <p style="font-size:0.9rem; opacity:0.85;">An internal evaluator grades retrieved chunks for relevance, filters out noise, and rewrites queries to eliminate hallucinations.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    meta = st.session_state.doc_metadata
    crag_tag = '<span class="badge-crag">CRAG: Active</span>' if enable_crag else '<span class="badge">Standard RAG</span>'
    if provider_key == "ollama":
        prov_tag = f'<span class="badge-ollama">Ollama: {model_name}</span>'
    elif provider_key == "groq":
        prov_tag = f'<span class="badge-groq">Groq: {model_name}</span>'
    else:
        prov_tag = f'<span class="badge">Gemini: {model_name}</span>'
    
    st.markdown(
        f"""
        <div class="doc-banner">
            <div>
                <b>Active Library:</b> <code>{meta['total_books']} Document(s) Loaded</code>
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

    with st.expander("🗺️ How LangGraph CRAG Works (Agentic State Flow)", expanded=False):
        st.markdown("""
        **Stateful LangGraph Workflow (`rag_engine/crag_graph.py`):**
        ```mermaid
        graph LR
            A[User Query] --> B[Node: retrieve]
            B --> C[Node: grade_documents]
            C -->|Relevant Context >= 1| D[Node: generate]
            C -->|Noisy Context / Low Score| E[Node: rewrite_query]
            E --> B
            D --> F[Final Answer + Page Citations]
        ```
        - **Node `retrieve`**: Pulls top-$k$ semantic chunks from the ChromaDB vector store.
        - **Node `grade_documents`**: Evaluates each retrieved passage for relevance (`yes`/`no`), stripping out noise.
        - **Conditional Edge (`decide_to_generate`)**: If relevant chunks are found, routes directly to `generate`. If chunks are irrelevant, triggers `rewrite_query` to reformulate the search query and re-retrieves.
        - **Node `generate`**: Streams the final grounded answer with page citations.
        """)

    # Suggested Prompts
    if len(st.session_state.chat_history) == 0:
        st.markdown("##### 💡 Suggested Questions across Library:")
        quick_cols = st.columns(3)
        with quick_cols[0]:
            if st.button("📌 Summarize key concepts & takeaways", use_container_width=True):
                st.session_state.quick_query = "Provide a comprehensive summary of the key concepts, main arguments, and conclusions across the uploaded documents."
        with quick_cols[1]:
            if st.button("🔑 Compare topics across documents", use_container_width=True):
                st.session_state.quick_query = "Compare and contrast the perspectives or rules presented across the different uploaded documents."
        with quick_cols[2]:
            if st.button("📋 Extract actionable steps & rules", use_container_width=True):
                st.session_state.quick_query = "Extract all actionable guidelines, requirements, and key recommendations mentioned in the text."

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("crag_info"):
                ci = msg["crag_info"]
                st.caption(f"🔍 **CRAG Evaluation:** Evaluated {ci.get('initial_retrieved', 0)} chunks → Kept {ci.get('relevant_filtered', 0)} relevant" + (f" | Query Rewritten: *{ci.get('rewritten_query')}*" if ci.get('query_rewritten') else ""))
            if msg.get("sources"):
                with st.expander(f"📖 View Referenced Sources ({len(msg['sources'])} chunks across documents)", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        page_num = src.get("page", "Unknown")
                        chunk_id = src.get("chunk_id", idx + 1)
                        source_file = src.get("source", "Document")
                        snippet = src.get("content", "")
                        st.markdown(f"**Source #{idx+1} — `{source_file}` | Page {page_num} (Chunk {chunk_id})**")
                        st.markdown(f'<div class="source-box">{snippet}</div>', unsafe_allow_html=True)

    # Handle User Query
    user_query = st.chat_input("Ask any question across your uploaded documents...")

    if "quick_query" in st.session_state and st.session_state.quick_query:
        user_query = st.session_state.quick_query
        st.session_state.quick_query = None

    if user_query:
        if provider_key == "gemini" and not api_key:
            st.error("⚠️ Please enter a Google API Key in the sidebar.")
        elif provider_key == "groq" and not api_key:
            st.error("⚠️ Please enter a Groq API Key in the sidebar (get a free key at https://console.groq.com/).")
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
                    with st.spinner("⚡ Executing LangGraph Agentic CRAG Workflow (Retrieve ➔ Grade ➔ Generate)..."):
                        token_stream, source_docs, crag_stats = stream_langgraph_crag_pipeline(
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
                        st.caption(f"🧠 **LangGraph Flow:** Node `retrieve` ({crag_stats['initial_retrieved']} chunks) ➔ Node `grade_documents` ({crag_stats['relevant_filtered']} kept)" + (f" ➔ Node `rewrite_query` (*{crag_stats['rewritten_query']}*)" if crag_stats['query_rewritten'] else "") + " ➔ Node `generate`")
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
                    with st.expander(f"📖 View Referenced Sources ({len(sources_data)} chunks across documents)", expanded=False):
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
        chat_export_text = "# DocuQuery AI - Multi-Document Conversation History\n\n"
        if st.session_state.doc_metadata:
            chat_export_text += f"**Library:** {st.session_state.doc_metadata.get('total_books')} documents ({st.session_state.doc_metadata.get('total_pages')} total pages)\n\n---\n\n"
        for msg in st.session_state.chat_history:
            speaker = "🧑 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
            chat_export_text += f"### {speaker}\n{msg['content']}\n\n"
        
        st.download_button(
            label="📥 Download Conversation History (Markdown)",
            data=chat_export_text,
            file_name="conversation_history.md",
            mime="text/markdown",
            use_container_width=True,
        )
