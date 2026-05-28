"""Streamlit UI for the RAG system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.rag.pipeline import RAGPipeline

st.set_page_config(page_title="AI Knowledge Base Q&A", page_icon="🔍", layout="wide")
st.title("AI Knowledge Base Q&A")
st.markdown("Ask questions about AI topics from the knowledge base. Answers are grounded with citations.")


@st.cache_resource
def load_pipeline():
    try:
        return RAGPipeline()
    except FileNotFoundError:
        return None


pipeline = load_pipeline()

if pipeline is None:
    st.error("Vector index not found. Please run: `python -m src.indexing.build_index`")
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of sources (top_k)", min_value=1, max_value=15, value=5)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This system uses RAG to provide grounded answers from a domain-specific knowledge base.")

question = st.text_input("Enter your question:", placeholder="What is discussed about AI safety?")

col1, col2 = st.columns([1, 1])
ask_clicked = col1.button("Ask", type="primary")
clear_clicked = col2.button("Clear")

if clear_clicked:
    st.session_state.pop("response", None)
    st.rerun()

if ask_clicked and question:
    with st.spinner("Retrieving relevant context and generating answer..."):
        st.session_state["response"] = pipeline.ask(question, top_k=top_k)

if "response" in st.session_state:
    response = st.session_state["response"]

    st.markdown("## Answer")
    st.markdown(response.answer)

    st.markdown("---")
    st.markdown("## Sources & Citations")
    for i, citation in enumerate(response.citations, 1):
        with st.expander(f"Source {i}: {citation['title']} (score: {citation['score']:.3f})"):
            st.markdown(f"**URL:** {citation['source_url']}")
            st.markdown(f"**Section:** {citation['section']}")
            chunk = response.retrieved_contexts[i - 1] if i - 1 < len(response.retrieved_contexts) else {}
            if chunk:
                st.markdown("**Retrieved text:**")
                st.text(chunk.get("text", "")[:500])

    st.markdown("---")
    st.markdown("## Retrieval Details")
    scores = [c["score"] for c in response.citations]
    if scores:
        st.bar_chart({"Relevance Score": scores})
