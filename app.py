import streamlit as st
from rag_engine import (
    add_document,
    get_documents_list,
    delete_document,
    get_doc_stats,
    semantic_search,
    answer_question,
    collection
)

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📑")
st.title("📑 Multi-PDF RAG System")
st.caption("Upload multiple PDFs, ask questions across all of them")

# sidebar
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("➕ Add All to Knowledge Base"):
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    msg = add_document(file)
                st.write(msg)
            st.rerun()

    st.divider()

    # document list with stats
    st.header("📄 Knowledge Base")
    docs = get_documents_list()

    if docs:
        for doc in docs:
            chunks = get_doc_stats(doc)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📄 {doc}")
                st.caption(f"   {chunks} chunks")
            with col2:
                if st.button("🗑️", key=f"del_{doc}"):
                    delete_document(doc)
                    st.rerun()
    else:
        st.info("No documents yet.")

    st.divider()
    st.metric("Total Chunks", collection.count())
    st.metric("Total Documents", len(docs))
    st.caption("Day 13 — 30 Day AI Bootcamp")

# main area
docs = get_documents_list()

# document selector
st.subheader("🔎 Select Documents to Search")
if docs:
    selected_docs = st.multiselect(
        "Choose documents (empty = search all)",
        options=docs,
        default=[]
    )
else:
    selected_docs = []
    st.info("👆 Upload documents from the sidebar to get started.")

# mode selector
mode = st.radio(
    "Mode:",
    ["💬 Q&A", "⚖️ Compare Documents"],
    horizontal=True
)
compare_mode = mode == "⚖️ Compare Documents"

st.divider()

# tabs
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Raw Search"])

with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    placeholder = (
        "Compare these documents..."
        if compare_mode
        else "Ask anything about your documents..."
    )
    question = st.chat_input(placeholder)

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Searching across documents..."):
                results = semantic_search(
                    question,
                    selected_docs=selected_docs if selected_docs else None
                )
                answer = answer_question(
                    question,
                    results,
                    mode="compare" if compare_mode else "qa"
                )
            st.write(answer)

            if results:
                with st.expander("📌 Sources"):
                    for doc, meta, distance in results:
                        similarity = round((1 - distance) * 100, 1)
                        st.caption(
                            f"📄 {meta['filename']} | "
                            f"Chunk {meta['chunk']} | "
                            f"{similarity}% match"
                        )
                        st.text(doc[:150] + "...")
                        st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

with tab2:
    st.subheader("Raw Semantic Search")
    raw_query = st.text_input("Search chunks directly...")

    if raw_query:
        results = semantic_search(
            raw_query,
            selected_docs=selected_docs if selected_docs else None
        )
        if results:
            for i, (doc, meta, distance) in enumerate(results):
                similarity = round((1 - distance) * 100, 1)
                with st.expander(
                    f"Result {i+1} — {meta['filename']} "
                    f"| {similarity}% match"
                ):
                    st.write(doc)
        else:
            st.info("No results found.")