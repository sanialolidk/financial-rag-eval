"""
Streamlit Cloud entrypoint — FinRAG Eval live demo.
Deploy: connect repo on share.streamlit.io, main file = app.py
Secrets: OPENAI_API_KEY (required), SEC_USER_AGENT (optional)
"""

from __future__ import annotations

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Inject Streamlit secrets into environment before config loads
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "SEC_USER_AGENT" in st.secrets:
        os.environ["SEC_USER_AGENT"] = st.secrets["SEC_USER_AGENT"]
except Exception:
    pass

st.set_page_config(page_title="FinRAG Eval", page_icon="📊", layout="wide")

st.title("FinRAG Eval — Financial Document RAG")
st.caption("SEC 10-K / 10-Q · hybrid retrieval · cross-encoder rerank · faithfulness eval")

EXAMPLES = [
    "What Data Center revenue does NVIDIA report?",
    "What gross margin does Apple report for its products?",
    "What does JPMorgan report about net interest income?",
    "What was Apple's Q3 2030 holographic display revenue guidance?",  # should refuse
]


@st.cache_resource(show_spinner="Loading retrieval indexes and models…")
def _pipeline():
    from app.pipeline.query import RAGPipeline
    return RAGPipeline()


with st.sidebar:
    st.header("Demo")
    st.markdown(
        "**Corpus:** 8 companies · 16 SEC filings · 3,013 chunks  \n"
        "**Retrieval:** Chroma + BM25 → cross-encoder rerank"
    )
    run_eval = st.checkbox("Run eval scores", value=True)
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set `OPENAI_API_KEY` in Streamlit secrets to enable answers.")

col_ex, col_q = st.columns([1, 2])
with col_ex:
    st.subheader("Examples")
    picked = st.radio("Pick a question", EXAMPLES, label_visibility="collapsed")
with col_q:
    question = st.text_area("Your question", value=picked, height=90)

if st.button("Ask", type="primary") and question.strip():
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Add OPENAI_API_KEY in app Settings → Secrets to query.")
        st.stop()
    try:
        pipeline = _pipeline()
    except Exception as exc:
        st.error(f"Pipeline failed to load: {exc}")
        st.stop()

    with st.spinner("Retrieving → reranking → generating → evaluating…"):
        try:
            result = pipeline.query(question.strip(), run_eval=run_eval)
            data = result.to_dict()
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{data.get('confidence', 0):.2f}")
    lat = data.get("latency", {})
    m2.metric("Retrieval", f"{lat.get('retrieval_ms', 0) + lat.get('rerank_ms', 0):.0f} ms")
    m3.metric("Total", f"{lat.get('total_ms', 0):.0f} ms")
    m4.metric("Refused", "Yes" if data.get("refused") else "No")

    if data.get("refused"):
        st.warning(data.get("refusal_reason", "Answer refused"))

    st.subheader("Answer")
    st.markdown(data.get("answer", ""))

    scores = data.get("scores")
    if scores:
        st.subheader("Eval scores")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
        c2.metric("Chunk recall", f"{scores.get('chunk_recall', 0):.2f}")
        c3.metric("Relevance", f"{scores.get('response_relevance', 0):.2f}")
        c4.metric("Hallucination", "Yes" if scores.get("hallucination_detected") else "No")
        for alert in scores.get("alerts", []):
            st.error(alert)

    st.subheader("Top cited chunks")
    for chunk in data.get("chunks", [])[:5]:
        with st.expander(
            f"{chunk.get('citation', chunk.get('chunk_id'))} "
            f"— relevance {chunk.get('chunk_relevance_score', 0):.2f}"
        ):
            st.text(chunk.get("text_preview", ""))

st.divider()
st.caption(
    "[GitHub](https://github.com/sanialolidk/financial-rag-eval) · "
    "Built with hybrid RAG + RAGAS-style eval harness"
)