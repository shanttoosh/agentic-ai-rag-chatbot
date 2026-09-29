"""Streamlit UI; calls the FastAPI backend (POST /api/chat) over HTTP.

    streamlit run ui/streamlit_app.py

Set API_URL to point at the backend (default http://localhost:8000).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 180.0  # retrieval + generate + validate

SAMPLE_QUESTIONS = (
    "What is Agentic AI?",
    "What are the defining characteristics of an AI agent?",
    "How are AI agents different from LLMs?",
    "How does an agentic AI system decide which action to take?",
    "What are the challenges of orchestrating multi-agent systems?",
    "How should an organization get started with implementing Agentic AI?",
    "What is the capital of France?",
)

STATUS_EXPLANATIONS = {
    "answered": "The answer passed the grounding validation against the retrieved chunks.",
    "low_retrieval_score": "No chunk cleared the retrieval threshold, so the LLM was never called.",
    "not_in_context": "Chunks were retrieved, but none of them answers this question.",
    "failed_validation": "The draft answer made claims the retrieved chunks don't support, so it "
    "was withheld.",
    "declined": "The model declined to answer this question.",
}


class ApiError(Exception):
    """The backend could not be reached or returned an error."""


# ----------------------------------------------------------------------- API client


def ask_api(api_url: str, question: str, top_k: int) -> dict[str, Any]:
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/api/chat",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.ConnectError as exc:
        raise ApiError(
            f"Can't reach the API at {api_url}. Start it with `uvicorn app.main:app`."
        ) from exc
    except httpx.TimeoutException as exc:
        raise ApiError("The API took too long to answer. Try again.") from exc

    if response.status_code != 200:
        raise ApiError(_error_message(response))
    result: dict[str, Any] = response.json()
    return result


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"API error (HTTP {response.status_code})"
    if isinstance(body, dict) and "message" in body:
        return f"{body['message']} (HTTP {response.status_code})"
    if isinstance(body, dict) and "detail" in body:  # FastAPI request validation error
        return f"Invalid request: {body['detail']}"
    return f"API error (HTTP {response.status_code})"


def api_is_healthy(api_url: str) -> bool:
    try:
        return httpx.get(f"{api_url.rstrip('/')}/health", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


# ------------------------------------------------------------------------ rendering


def render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        top_k = st.slider("Chunks to retrieve (top-K)", min_value=1, max_value=10, value=5)
        if api_is_healthy(api_url):
            st.success("API is up", icon="✅")
        else:
            st.error("API is not reachable", icon="⚠️")

        st.header("Try a question")
        for sample in SAMPLE_QUESTIONS:
            if st.button(sample, width="stretch"):
                st.session_state.question = sample
                st.session_state.submit = True

        st.header("How it works")
        st.markdown(
            "1. The question is embedded and searched in **Pinecone**.\n"
            "2. Chunks below the similarity threshold are dropped.\n"
            "3. The **LLM** (Groq gpt-oss) answers from those chunks only.\n"
            "4. A validator checks every claim against those chunks.\n"
            "5. Anything unsupported returns the safe fallback."
        )
    return api_url, top_k


def render_answer(result: dict[str, Any]) -> None:
    st.subheader("Answer")
    with st.container(border=True):
        st.markdown(result["answer"])
        if result["cited_pages"]:
            pages = ", ".join(str(page) for page in result["cited_pages"])
            st.caption(f"Source: eBook page{'s' if len(result['cited_pages']) > 1 else ''} {pages}")

    details = result["confidence_details"]
    confidence, grounded, top, relevant = st.columns(4)
    confidence.metric(
        "Retrieval confidence", f"{result['confidence']:.2f}", details["level"], delta_color="off"
    )
    grounded.metric("Grounded", "✓ Yes" if result["grounded"] else "✗ No")
    top.metric("Top similarity", f"{details['top_score']:.3f}")
    relevant.metric(
        "Chunks above threshold", f"{details['relevant_chunks']} / {details['retrieved_chunks']}"
    )

    explanation = STATUS_EXPLANATIONS.get(result["status"], result["status"])
    (st.success if result["grounded"] else st.info)(explanation)
    st.caption(
        "Confidence is a heuristic built from the similarity scores (top score, mean top-K "
        "score, number of relevant chunks). It is not a calibrated probability."
    )


def render_context(result: dict[str, Any]) -> None:
    chunks = result["retrieved_context"]
    with st.expander(f"Retrieved context: {len(chunks)} chunks from Pinecone", expanded=True):
        st.caption(
            "This is the evidence the answer was generated from. Only chunks above the "
            "threshold were sent to the LLM."
        )
        for chunk in chunks:
            tags = []
            if chunk["cited"]:
                tags.append(":green[**cited**]")
            tags.append(
                ":blue[sent to LLM]" if chunk["above_threshold"] else ":gray[below threshold]"
            )
            section = f" · {chunk['section']}" if chunk.get("section") else ""
            st.markdown(
                f"**Page {chunk['page']}** · score **{chunk['score']:.3f}**{section} · "
                f"{' · '.join(tags)}  \n`{chunk['chunk_id']}`"
            )
            with st.container(border=True):
                st.markdown(chunk["text"].replace("\n", "  \n"))

    with st.expander("Raw API response"):
        st.json(result)


# ----------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="Agentic AI Knowledge Assistant", page_icon="📘", layout="wide")
    st.title("📘 Agentic AI Knowledge Assistant")
    st.caption("Grounded exclusively in the *Agentic AI* eBook · LangGraph + Pinecone + Groq")

    api_url, top_k = render_sidebar()

    with st.form("ask", clear_on_submit=False):
        question = st.text_input(
            "Ask a question", key="question", placeholder="e.g. What is Agentic AI?"
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted or st.session_state.pop("submit", False):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving from Pinecone and generating a grounded answer..."):
                try:
                    st.session_state.result = ask_api(api_url, question, top_k)
                except ApiError as exc:
                    st.session_state.result = None
                    st.error(str(exc))

    result = st.session_state.get("result")
    if result:
        render_answer(result)
        render_context(result)


main()
