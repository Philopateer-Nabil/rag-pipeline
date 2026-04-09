"""Streamlit chat UI for the RAG pipeline."""

from __future__ import annotations

import streamlit as st

from rag_pipeline.config import load_settings
from rag_pipeline.pipeline import RAGPipeline


def main():
    st.set_page_config(page_title="RAG Pipeline", page_icon="🔍", layout="wide")

    st.title("RAG Pipeline")
    st.caption("Ask questions about AI/ML concepts — powered by local models")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        config_path = st.text_input("Config file", value="config.yaml")

        settings = load_settings(config_path)

        settings.retrieval.strategy = st.selectbox(
            "Retrieval strategy",
            ["mmr", "similarity", "hybrid"],
            index=["mmr", "similarity", "hybrid"].index(settings.retrieval.strategy)
            if settings.retrieval.strategy in ("mmr", "similarity", "hybrid")
            else 0,
        )
        settings.retrieval.top_k = st.slider("Retrieval top-k", 1, 20, settings.retrieval.top_k)
        settings.reranker.enabled = st.checkbox("Enable re-ranking", settings.reranker.enabled)
        if settings.reranker.enabled:
            settings.reranker.top_k = st.slider(
                "Reranker top-k", 1, 10, settings.reranker.top_k
            )
        settings.generation.model_name = st.selectbox(
            "LLM model", ["llama3", "mistral"], index=0
        )
        settings.generation.temperature = st.slider(
            "Temperature", 0.0, 1.0, settings.generation.temperature
        )

        st.divider()

        # Ingest section
        st.header("Ingest Documents")
        ingest_path = st.text_input("Directory or file path", value="data/sample/")
        if st.button("Ingest", type="primary"):
            with st.spinner("Ingesting..."):
                try:
                    pipeline = _get_pipeline(settings)
                    n = pipeline.ingest(ingest_path)
                    st.success(f"Ingested {n} chunks")
                    # Force pipeline reload on next query
                    st.session_state.pop("pipeline_id", None)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

        st.divider()

        # Metrics section
        st.header("Pipeline Metrics")
        if st.button("Refresh Metrics"):
            st.session_state["show_metrics"] = True

        if st.session_state.get("show_metrics"):
            from rag_pipeline.metrics import collector

            snapshot = collector.snapshot()
            st.metric("Total queries", snapshot["total_queries"])
            st.metric("Chunks ingested", snapshot["total_chunks_ingested"])
            st.metric("Est. tokens used", snapshot["estimated_tokens_used"])
            if snapshot["stages"]:
                st.subheader("Stage Latency (ms)")
                for stage, data in snapshot["stages"].items():
                    st.text(
                        f"{stage}: avg={data['avg_latency_ms']:.0f}  "
                        f"min={data['min_latency_ms']:.0f}  "
                        f"max={data['max_latency_ms']:.0f}  "
                        f"({data['total_calls']} calls)"
                    )

    # --- Main chat area ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {src['source']}**")
                        st.code(src["content"], language=None)
            if msg.get("metrics"):
                m = msg["metrics"]
                cols = st.columns(4)
                cols[0].metric("Retrieved", m.get("retrieved", "—"))
                cols[1].metric("After rerank", m.get("reranked", "—"))
                cols[2].metric("Latency", f"{m.get('total_ms', 0):.0f}ms")
                cols[3].metric("Est. tokens", m.get("tokens", "—"))

    # Chat input
    if question := st.chat_input("Ask a question about AI/ML..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"), st.spinner("Thinking..."):
                try:
                    pipeline = _get_pipeline(settings)
                    result = pipeline.query(question)

                    st.markdown(result.answer)

                    sources = [
                        {
                            "source": doc.metadata.get("source", "unknown"),
                            "content": doc.page_content[:400],
                        }
                        for doc in result.source_documents
                    ]
                    with st.expander(f"Sources ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**[{i}] {src['source']}**")
                            st.code(src["content"], language=None)

                    # Show query metrics
                    from rag_pipeline.metrics import collector

                    snap = collector.snapshot()
                    total_ms = sum(
                        s["avg_latency_ms"] * s["total_calls"]
                        for s in snap["stages"].values()
                    )
                    metrics_data = {
                        "retrieved": result.num_chunks_retrieved,
                        "reranked": result.num_chunks_after_rerank,
                        "total_ms": total_ms / max(snap["total_queries"], 1),
                        "tokens": snap["estimated_tokens_used"],
                    }
                    cols = st.columns(4)
                    cols[0].metric("Retrieved", metrics_data["retrieved"])
                    cols[1].metric("After rerank", metrics_data["reranked"])
                    cols[2].metric("Latency", f"{metrics_data['total_ms']:.0f}ms")
                    cols[3].metric("Est. tokens", metrics_data["tokens"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": sources,
                        "metrics": metrics_data,
                    })

                except FileNotFoundError:
                    err = "No index found. Please ingest documents first (sidebar)."
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err,
                    })
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err,
                    })


def _get_pipeline(settings) -> RAGPipeline:
    """Cache the pipeline in session state, rebuild if settings change."""
    settings_id = settings.model_dump_json()
    if (
        "pipeline" not in st.session_state
        or st.session_state.get("pipeline_id") != settings_id
    ):
        st.session_state.pipeline = RAGPipeline(settings=settings)
        st.session_state.pipeline_id = settings_id
    return st.session_state.pipeline


if __name__ == "__main__":
    main()
