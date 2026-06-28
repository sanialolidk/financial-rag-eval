"""Streamlit frontend — queries FastAPI, shows answers, chunks, and eval scores."""

from __future__ import annotations

import os
import sys

import httpx
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FinRAG Eval",
    page_icon="📊",
    layout="wide",
)

st.title("Financial Document RAG + Eval Harness")
st.caption("SEC 10-K / 10-Q hybrid retrieval with faithfulness scoring")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", API_BASE)
    run_eval = st.checkbox("Run eval scores (LLM-as-judge)", value=True)
    if st.button("Health check"):
        try:
            r = httpx.get(f"{api_base}/health", timeout=10.0)
            st.json(r.json())
        except Exception as exc:
            st.error(str(exc))

tab_query, tab_eval, tab_logs = st.tabs(["Ask", "Run Eval Set", "Query Logs"])

with tab_query:
    question = st.text_area(
        "Question",
        placeholder="e.g. What gross margin does NVIDIA report?",
        height=100,
    )
    if st.button("Submit", type="primary") and question.strip():
        with st.spinner("Retrieving, reranking, generating..."):
            try:
                resp = httpx.post(
                    f"{api_base}/query",
                    json={"question": question.strip(), "run_eval": run_eval},
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{data.get('confidence', 0):.2f}")
        lat = data.get("latency", {})
        col2.metric("Total latency", f"{lat.get('total_ms', 0):.0f} ms")
        cost = data.get("cost", {})
        col3.metric("Cost", f"${cost.get('total_cost_usd', 0):.5f}")

        if data.get("refused"):
            st.warning(data.get("refusal_reason", "Refused"))

        st.subheader("Answer")
        st.markdown(data.get("answer", ""))

        scores = data.get("scores")
        if scores:
            st.subheader("Eval scores")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
            s2.metric("Chunk recall", f"{scores.get('chunk_recall', 0):.2f}")
            s3.metric("Response relevance", f"{scores.get('response_relevance', 0):.2f}")
            s4.metric("Hallucination", "Yes" if scores.get("hallucination_detected") else "No")
            if scores.get("alerts"):
                for alert in scores["alerts"]:
                    st.error(alert)

        st.subheader("Cited chunks")
        for chunk in data.get("chunks", []):
            with st.expander(f"{chunk.get('citation', chunk.get('chunk_id'))} — rel {chunk.get('chunk_relevance_score', 0):.2f}"):
                st.text(chunk.get("text_preview", ""))

        with st.expander("Latency breakdown"):
            st.json(lat)

with tab_eval:
    limit = st.number_input("Eval queries (max)", min_value=1, max_value=22, value=5)
    if st.button("Run eval harness"):
        with st.spinner(f"Running {limit} eval queries..."):
            try:
                resp = httpx.post(
                    f"{api_base}/eval/run",
                    json={"limit": int(limit)},
                    timeout=600.0,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        st.subheader("Summary")
        st.json(payload.get("summary", {}))
        st.subheader("Per-query results")
        for r in payload.get("results", []):
            scores = r.get("scores") or {}
            st.markdown(
                f"**{r.get('query', '')[:80]}** — "
                f"faith {scores.get('faithfulness', 0):.2f}, "
                f"recall {scores.get('chunk_recall', 0):.2f}, "
                f"refused={r.get('refused')}"
            )

with tab_logs:
    try:
        resp = httpx.get(f"{api_base}/metrics", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        st.subheader("Latest eval summary")
        st.json(data.get("latest_eval_summary"))
        st.subheader("Recent queries")
        st.dataframe(data.get("recent_queries", []), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))