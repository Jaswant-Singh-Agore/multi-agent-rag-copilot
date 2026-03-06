"""
Streamlit chat interface for the Multi-Agent RAG Copilot.
"""

import requests
import streamlit as st


API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Agent RAG Copilot",
    page_icon="🤖",
    layout="wide",
)


def check_api() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def upload_files(files) -> dict:
    file_data = [
        ("files", (f.name, f.getvalue(), "application/pdf"))
        for f in files
    ]
    try:
        r = requests.post(f"{API_BASE}/upload", files=file_data, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_api(question: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"answer": f"Something went wrong: {e}"}


def show_metadata(result: dict) -> None:
    confidence = result.get("confidence", "N/A")
    is_grounded = result.get("is_grounded", False)

    col1, col2 = st.columns(2)
    col1.metric("Confidence", confidence)
    col2.metric("Grounded", "Yes" if is_grounded else "No")

    entities = result.get("graph_entities", [])
    if entities:
        st.caption(f"Graph entities: {', '.join(entities)}")

    sources = result.get("sources", [])
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.write(f"**{s.get('source')}** — Page {s.get('page')} (score: {s.get('score', 0):.3f})")


with st.sidebar:
    st.title("RAG Copilot")

    health = check_api()
    if health.get("status") == "healthy":
        st.success("API connected")
        vector_count = health.get("faiss_vectors", 0)
        st.info(f"{vector_count} chunk(s) indexed")
    else:
        st.error("API not reachable — is the server running?")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Index Documents", use_container_width=True):
            with st.spinner("Processing and indexing..."):
                result = upload_files(uploaded_files)

            if result.get("status") == "success":
                st.success(f"Indexed {result.get('total_chunks', 0)} chunk(s).")
                st.session_state.messages = []
                st.rerun()
            else:
                st.error(result.get("message", "Upload failed."))

    st.divider()

    if st.button("Reset Knowledge Base", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/reset", timeout=10)
            st.success("Knowledge base cleared.")
            st.session_state.messages = []
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")


st.title("Multi-Agent RAG Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = query_api(question)

        answer = result.get("answer", "No answer returned.")
        st.markdown(answer)
        show_metadata(result)

    st.session_state.messages.append({"role": "assistant", "content": answer})