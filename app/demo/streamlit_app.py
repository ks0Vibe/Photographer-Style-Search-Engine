from __future__ import annotations

from html import escape
import os

import requests
import streamlit as st


PRESET_QUERIES = [
    "warm cinematic landscape",
    "dark moody forest",
    "person in street photography",
    "car at night",
    "dog on beach",
    "bird in nature",
    "minimal architecture",
]


THEME_CSS = """
<style>
    :root {
        --wine: #531426;
        --wine-deep: #42101e;
        --wine-card: #74243d;
        --cream: #eaded0;
        --cream-soft: #f6eee7;
        --gold: #d4ad5d;
        --ink: #531426;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--wine);
        color: var(--cream-soft);
    }

    [data-testid="stHeader"] { background: transparent; }

    [data-testid="stSidebar"] {
        background: var(--wine-deep);
        border-right: 1px solid rgba(212, 173, 93, 0.55);
    }

    [data-testid="stSidebar"] * { color: var(--cream-soft); }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(246, 238, 231, 0.78);
        line-height: 1.45;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--cream-soft) !important;
        letter-spacing: -0.03em;
    }

    .hero {
        border-bottom: 2px solid var(--gold);
        padding: 1.25rem 0 1.4rem;
        margin-bottom: 1.35rem;
    }

    .hero-eyebrow,
    .section-label {
        color: var(--gold);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    .hero-eyebrow { margin-bottom: 0.55rem; }

    .hero h1 {
        color: var(--cream-soft);
        font-size: clamp(2.25rem, 5vw, 4.6rem);
        line-height: 0.98;
        margin: 0;
    }

    .hero p {
        color: rgba(246, 238, 231, 0.82);
        font-size: 1.05rem;
        margin: 0.9rem 0 0;
    }

    .hero-meta {
        color: var(--gold);
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        margin-top: 1rem;
    }

    .section-label { margin: 0.2rem 0 0.65rem; }

    .signal-card,
    .query-help {
        background: var(--wine-card);
        border: 1px solid rgba(212, 173, 93, 0.42);
        border-radius: 1rem;
        padding: 1rem 1.15rem;
        color: var(--cream-soft);
    }

    .signal-card strong,
    .query-help strong { color: var(--gold); }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.7rem;
        border-bottom: 1px solid rgba(212, 173, 93, 0.45);
    }

    .stTabs [data-baseweb="tab"] {
        color: rgba(246, 238, 231, 0.66);
        font-weight: 700;
        padding: 0.75rem 1.1rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--cream-soft) !important;
        border-bottom-color: var(--gold) !important;
    }

    .stTextInput label,
    .stNumberInput label,
    .stCheckbox label,
    .stSlider label,
    .stFileUploader label {
        color: var(--cream-soft) !important;
        font-weight: 600;
    }

    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label p {
        color: var(--cream-soft) !important;
        opacity: 1 !important;
    }

    [data-testid="stCheckbox"] input {
        accent-color: var(--gold);
    }

    [data-testid="stCheckbox"] div[role="checkbox"] {
        background: transparent;
        border: 1px solid var(--gold);
        border-radius: 0.25rem;
    }

    [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
        background: var(--gold);
        border-color: var(--gold);
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="textarea"] {
        background: var(--cream-soft);
        border-radius: 0.55rem;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] * { color: var(--ink) !important; }

    div.stButton > button {
        background: transparent;
        border: 1px solid rgba(212, 173, 93, 0.62);
        border-radius: 0.65rem;
        color: var(--cream-soft);
        font-weight: 650;
        min-height: 2.7rem;
        transition: all 0.18s ease;
    }

    div.stButton > button:hover {
        background: rgba(212, 173, 93, 0.16);
        border-color: var(--gold);
        color: var(--cream-soft);
    }

    div.stButton > button[kind="primary"] {
        background: var(--gold);
        border-color: var(--gold);
        color: var(--wine-deep);
        font-size: 1rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background: var(--cream-soft);
        border-color: var(--cream-soft);
        color: var(--wine-deep);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(116, 36, 61, 0.76);
        border: 1px solid rgba(212, 173, 93, 0.34);
        border-radius: 1rem;
        padding: 0.55rem;
    }

    [data-testid="stImage"] img { border-radius: 0.7rem; }

    .result-rank {
        color: var(--gold);
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.8rem;
    }

    .result-score {
        color: var(--cream-soft);
        font-size: 1.15rem;
        font-weight: 750;
        margin: 0.2rem 0 0.45rem;
    }

    .result-description {
        color: rgba(246, 238, 231, 0.88);
        line-height: 1.45;
        min-height: 2.8rem;
    }

    .result-meta {
        color: rgba(246, 238, 231, 0.68);
        font-size: 0.78rem;
        line-height: 1.5;
        margin-top: 0.7rem;
    }

    [data-testid="stAlert"] { border-radius: 0.8rem; }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: rgba(246, 238, 231, 0.66) !important;
    }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="DeepStyle / Photographer Search", layout="wide")
    inject_theme()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">DeepStyle / visual search engine</div>
            <h1>Photographer Style Search</h1>
            <p>Find references by meaning, objects, mood and visual style.</p>
            <div class="hero-meta">CLIP / QDRANT / YOLO / STYLE-AWARE RERANKING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="section-label">Search signals</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="signal-card">
                <p><strong>Semantic</strong><br>CLIP meaning-based retrieval.</p>
                <p><strong>Keyword</strong><br>Unsplash metadata filters.</p>
                <p><strong>Object</strong><br>YOLO detected objects.</p>
                <p><strong>Style</strong><br>Visual descriptors and reranking.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Connection</div>', unsafe_allow_html=True)
        api_base_url = st.text_input(
            "API base URL",
            value=os.getenv("DEEPSTYLE_API_URL", "http://localhost:8001"),
        ).rstrip("/")

    if "query" not in st.session_state:
        st.session_state.query = "dog on beach"

    text_tab, image_tab = st.tabs(["Text search", "Image search"])

    with text_tab:
        render_text_search(api_base_url)

    with image_tab:
        render_image_search(api_base_url)


def render_text_search(api_base_url: str) -> None:
    st.markdown('<div class="section-label">Start with a visual idea</div>', unsafe_allow_html=True)
    preset_columns = st.columns(len(PRESET_QUERIES))
    for column, preset in zip(preset_columns, PRESET_QUERIES, strict=True):
        if column.button(preset):
            st.session_state.query = preset

    query = st.text_input("Describe the reference you want to find", key="query")

    control_columns = st.columns(4)
    top_k = control_columns[0].number_input("Results", min_value=1, max_value=50, value=10, step=1)
    candidate_pool_size = control_columns[1].number_input("Candidate pool", min_value=1, max_value=500, value=100, step=10)
    keyword = control_columns[2].text_input("Keyword filter", value="")
    requested_object = control_columns[3].text_input("Object filter", value="dog")

    rerank_columns = st.columns(3)
    rerank = rerank_columns[0].checkbox("Style reranking", value=False)
    object_rerank = rerank_columns[1].checkbox("Object reranking", value=True)
    enable_style_filters = rerank_columns[2].checkbox("Style filters", value=False)

    style_values = {}
    if enable_style_filters:
        style_columns = st.columns(4)
        for index, field in enumerate(("brightness", "contrast", "saturation", "warmth")):
            min_value, max_value = style_columns[index].slider(
                field,
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.01,
            )
            style_values[f"min_{field}"] = min_value
            style_values[f"max_{field}"] = max_value

    if st.button("Search references", type="primary"):
        body = {
            "query": query,
            "top_k": int(top_k),
            "candidate_pool_size": int(candidate_pool_size),
            "keyword": keyword.strip() or None,
            "object": requested_object.strip() or None,
            "rerank": bool(rerank),
            "object_rerank": bool(object_rerank),
            **style_values,
        }
        try:
            response = requests.post(f"{api_base_url}/search/text", json=body, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            render_request_error("Search failed", exc)
            return
        render_results(api_base_url, response.json())


def render_image_search(api_base_url: str) -> None:
    st.markdown('<div class="section-label">Use an existing image as a visual reference</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png", "webp"])
    image_id = st.text_input("Or use an image ID from the dataset", value="")

    control_columns = st.columns(3)
    top_k = control_columns[0].number_input("Results", min_value=1, max_value=50, value=10, step=1, key="image_top_k")
    candidate_pool_size = control_columns[1].number_input(
        "Candidate pool",
        min_value=1,
        max_value=500,
        value=100,
        step=10,
        key="image_candidate_pool_size",
    )
    rerank = control_columns[2].checkbox("Style reranking", value=False, key="image_style_rerank")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Query image", width=320)

    if st.button("Search similar images", type="primary"):
        if uploaded_file is not None:
            search_uploaded_image(api_base_url, uploaded_file, int(top_k), int(candidate_pool_size), bool(rerank))
            return

        if image_id.strip():
            search_dataset_image(api_base_url, image_id.strip(), int(top_k), int(candidate_pool_size), bool(rerank))
            return

        st.warning("Upload a query image or enter an image ID.")


def search_uploaded_image(
    api_base_url: str,
    uploaded_file,
    top_k: int,
    candidate_pool_size: int,
    rerank: bool,
) -> None:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {
        "top_k": str(top_k),
        "candidate_pool_size": str(candidate_pool_size),
        "rerank": "true" if rerank else "false",
    }
    try:
        response = requests.post(f"{api_base_url}/search/image/upload", files=files, data=data, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        render_request_error("Image upload search failed", exc)
        return
    render_results(api_base_url, response.json())


def search_dataset_image(
    api_base_url: str,
    image_id: str,
    top_k: int,
    candidate_pool_size: int,
    rerank: bool,
) -> None:
    body = {
        "image_id": image_id,
        "top_k": top_k,
        "candidate_pool_size": candidate_pool_size,
        "rerank": rerank,
    }
    try:
        response = requests.post(f"{api_base_url}/search/image", json=body, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        render_request_error("Image search failed", exc)
        return
    render_results(api_base_url, response.json())


def render_results(api_base_url: str, payload: dict) -> None:
    st.markdown('<div class="section-label">Ranked references</div>', unsafe_allow_html=True)
    st.caption(
        f"mode={payload.get('mode')} | top_k={payload.get('top_k')} | "
        f"latency_ms={float(payload.get('latency_ms', 0.0)):.1f}"
    )
    results = payload.get("results", [])
    if not results:
        st.info("No results.")
        return

    columns = st.columns(3)
    for index, result in enumerate(results):
        with columns[index % 3]:
            with st.container(border=True):
                image_id = str(result.get("image_id", ""))
                safe_image_id = escape(image_id)
                try:
                    st.image(f"{api_base_url}/image-file/{image_id}", use_container_width=True)
                except Exception:
                    st.info("Image file could not be loaded.")
                st.markdown(
                    f'<div class="result-rank">#{escape(str(result.get("rank", "")))} / {safe_image_id}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="result-score">Score {float(result.get("score", 0.0)):.4f}</div>',
                    unsafe_allow_html=True,
                )
                description = escape(str(result.get("ai_description") or "No description available."))
                st.markdown(
                    f'<div class="result-description">{description}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class=\"result-meta\">"
                    + "keywords: " + escape(", ".join(str(value) for value in (result.get("keywords") or [])[:8]))
                    + "<br>objects: " + escape(", ".join(str(value) for value in (result.get("detected_objects") or [])))
                    + "<br>style: "
                    + f"brightness={_fmt(result.get('brightness'))}, "
                    + f"contrast={_fmt(result.get('contrast'))}, "
                    + f"saturation={_fmt(result.get('saturation'))}, "
                    + f"warmth={_fmt(result.get('warmth'))}"
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if result.get("photo_url"):
                    st.link_button("Open on Unsplash", result["photo_url"])


def render_request_error(prefix: str, exc: requests.RequestException) -> None:
    response = getattr(exc, "response", None)
    if response is None:
        st.error(f"{prefix}: API is not reachable at the configured URL.")
        return
    if response.status_code == 404:
        st.error(
            f"{prefix}: the configured URL is not the DeepStyle API. "
            "Run DeepStyle on port 8001 or change the API base URL in the sidebar."
        )
        return
    st.error(f"{prefix}: {response.status_code} {response.text}")


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
