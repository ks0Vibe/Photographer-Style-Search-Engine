from __future__ import annotations

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


def main() -> None:
    st.set_page_config(page_title="Photographer Style Search", layout="wide")
    st.title("Photographer Style Search")

    with st.sidebar:
        st.header("Search Signals")
        st.write("Semantic search uses CLIP.")
        st.write("Keyword filter uses Unsplash metadata.")
        st.write("Object filter uses YOLO detected_objects.")
        st.write("Style rerank uses visual descriptors.")
        st.write("Object rerank combines semantic, object, keyword, and style signals.")
        api_base_url = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")

    if "query" not in st.session_state:
        st.session_state.query = "dog on beach"

    preset_columns = st.columns(len(PRESET_QUERIES))
    for column, preset in zip(preset_columns, PRESET_QUERIES, strict=True):
        if column.button(preset):
            st.session_state.query = preset

    query = st.text_input("Text query", key="query")

    control_columns = st.columns(4)
    top_k = control_columns[0].number_input("top_k", min_value=1, max_value=50, value=10, step=1)
    candidate_pool_size = control_columns[1].number_input("candidate_pool_size", min_value=1, max_value=500, value=100, step=10)
    keyword = control_columns[2].text_input("keyword", value="")
    requested_object = control_columns[3].text_input("object", value="dog")

    rerank_columns = st.columns(3)
    rerank = rerank_columns[0].checkbox("Style rerank", value=False)
    object_rerank = rerank_columns[1].checkbox("Object rerank", value=True)
    enable_style_filters = rerank_columns[2].checkbox("Enable style filters", value=False)

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

    if st.button("Search", type="primary"):
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
            st.error(f"Search failed: {exc}")
            return
        render_results(api_base_url, response.json())


def render_results(api_base_url: str, payload: dict) -> None:
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
            image_id = result.get("image_id", "")
            st.image(f"{api_base_url}/image-file/{image_id}", use_container_width=True)
            st.markdown(f"**#{result.get('rank')}** `{image_id}`")
            st.write(f"score: {float(result.get('score', 0.0)):.4f}")
            st.write(result.get("ai_description") or "")
            st.caption("keywords: " + ", ".join((result.get("keywords") or [])[:12]))
            st.caption("objects: " + ", ".join(result.get("detected_objects") or []))
            st.caption(
                "style: "
                f"brightness={_fmt(result.get('brightness'))}, "
                f"contrast={_fmt(result.get('contrast'))}, "
                f"saturation={_fmt(result.get('saturation'))}, "
                f"warmth={_fmt(result.get('warmth'))}"
            )
            if result.get("photo_url"):
                st.link_button("Unsplash", result["photo_url"])


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
