from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.api.dependencies import MetadataLookup
from app.ml.clip_encoder import CLIPEncoder
from app.search import (
    FaissIndex,
    MetadataRepository,
    ObjectAwareReranker,
    QdrantRetrievalService,
    QdrantStore,
    RerankCandidate,
    StyleReranker,
    VectorStore,
)
from app.search.metadata_repository import ImageMetadata
from app.search.qdrant_config import get_qdrant_settings


DEFAULT_QUERIES_PATH = PROJECT_ROOT / "experiments" / "08_validation_set" / "queries.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "09_validation_results"

DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "flat.index"

SYSTEMS = (
    "faiss_baseline",
    "faiss_style_rerank",
    "qdrant_semantic",
    "qdrant_filtered",
    "qdrant_object_rerank",
)
SYSTEM_OUTPUTS = {
    "faiss_baseline": "results_faiss_baseline.csv",
    "faiss_style_rerank": "results_faiss_style_rerank.csv",
    "qdrant_semantic": "results_qdrant_semantic.csv",
    "qdrant_filtered": "results_qdrant_filtered.csv",
    "qdrant_object_rerank": "results_qdrant_object_rerank.csv",
}
RESULT_FIELDS = [
    "query_id",
    "query",
    "type",
    "system",
    "rank",
    "image_id",
    "score",
    "image_path",
    "matched_keywords",
    "detected_objects",
    "brightness",
    "contrast",
    "saturation",
    "warmth",
    "notes",
]

TEXT_QUERY_TYPES = {"semantic_text", "style_text", "object_text", "mixed_text"}
STYLE_FIELDS = ("brightness", "contrast", "saturation", "warmth")
OBJECT_TERMS = {
    "person",
    "dog",
    "car",
    "bicycle",
    "bird",
    "boat",
    "cup",
    "train",
    "cat",
    "bus",
    "truck",
    "horse",
    "animal",
}


@dataclass(frozen=True)
class ValidationQuery:
    query_id: str
    query: str
    query_type: str
    intended_system: str
    notes: str


@dataclass
class RunState:
    warnings: list[str] = field(default_factory=list)
    skipped_image_queries: list[str] = field(default_factory=list)
    systems_attempted: list[str] = field(default_factory=list)
    systems_completed: list[str] = field(default_factory=list)
    systems_skipped: dict[str, str] = field(default_factory=dict)
    rows_per_system: dict[str, int] = field(default_factory=dict)


class FaissValidationSearch:
    def __init__(
        self,
        *,
        clip_encoder: CLIPEncoder,
        candidate_pool_size: int,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.vector_store = VectorStore(EMBEDDINGS_PATH, IMAGE_IDS_PATH)
        self.metadata_repository = MetadataRepository(DATABASE_PATH)
        self.faiss_index = FaissIndex(INDEX_PATH)
        self.faiss_index.load()
        self.style_reranker = StyleReranker()
        self.candidate_pool_size = candidate_pool_size

    def search_text(
        self,
        query: str,
        *,
        top_k: int,
        rerank: bool,
    ) -> tuple[list[dict[str, Any]], str]:
        query_vector = self.clip_encoder.encode_text(query.strip())
        return self._search_vector(
            query_vector,
            query_metadata=build_text_query_metadata(query) if rerank else None,
            top_k=top_k,
            rerank=rerank,
        )

    def search_image(
        self,
        image_path: Path,
        *,
        top_k: int,
        rerank: bool,
    ) -> tuple[list[dict[str, Any]], str]:
        from PIL import Image
        from app.ml.visual_descriptor import VisualDescriptorExtractor

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            query_vector = self.clip_encoder.encode_image(rgb_image)
            descriptors = VisualDescriptorExtractor().extract(rgb_image)

        query_metadata = ImageMetadata(
            image_id="__query_image__",
            file_path=str(image_path),
            brightness=float(descriptors["brightness"]),
            contrast=float(descriptors["contrast"]),
            saturation=float(descriptors["saturation"]),
            warmth=float(descriptors["warmth"]),
            color_histogram=tuple(float(value) for value in descriptors["color_histogram"]),
        )
        return self._search_vector(
            query_vector,
            query_metadata=query_metadata if rerank else None,
            top_k=top_k,
            rerank=rerank,
        )

    def _search_vector(
        self,
        query_vector: np.ndarray,
        *,
        query_metadata: ImageMetadata | None,
        top_k: int,
        rerank: bool,
    ) -> tuple[list[dict[str, Any]], str]:
        image_ids = self.vector_store.get_image_ids()
        search_top_k = min(len(image_ids), max(top_k, self.candidate_pool_size) if rerank else top_k)
        scores, indices = self.faiss_index.search(query_vector, top_k=search_top_k)

        ranked_ids: list[str] = []
        ranked_scores: list[float] = []
        for score, index in zip(scores, indices, strict=True):
            if index < 0 or index >= len(image_ids):
                continue
            ranked_ids.append(str(image_ids[index]))
            ranked_scores.append(float(score))

        metadata_by_id = self.metadata_repository.get_many(ranked_ids)
        candidates: list[RerankCandidate] = []
        for image_id, score in zip(ranked_ids, ranked_scores, strict=True):
            metadata = metadata_by_id.get(image_id)
            if metadata is None:
                continue
            candidates.append(
                RerankCandidate(
                    metadata=metadata,
                    semantic_score=score,
                    final_score=score,
                )
            )

        note = ""
        if rerank:
            if query_metadata is None:
                note = "No reliable text style cues inferred; returned FAISS semantic order."
            else:
                candidates = self.style_reranker.rerank(query_metadata, candidates)
                note = "FAISS semantic candidates reranked with existing style reranker."

        rows = [
            {
                "image_id": candidate.metadata.image_id,
                "file_path": candidate.metadata.file_path,
                "score": candidate.score,
                "semantic_score": candidate.semantic_score,
                "style_score": candidate.style_score,
                "final_score": candidate.final_score,
            }
            for candidate in candidates[:top_k]
        ]
        return rows, note


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate top-k validation search results for manual relevance labeling."
    )
    parser.add_argument("--queries-path", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument(
        "--systems",
        type=str,
        default="all",
        help="Comma-separated systems or 'all'.",
    )
    return parser.parse_args()


def load_queries(path: Path) -> list[ValidationQuery]:
    if not path.exists():
        raise FileNotFoundError(f"Validation query file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "query", "type", "intended_system", "notes"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Validation query file is missing columns: {sorted(missing)}")
        return [
            ValidationQuery(
                query_id=str(row["query_id"]).strip(),
                query=str(row["query"]).strip(),
                query_type=str(row["type"]).strip(),
                intended_system=str(row["intended_system"]).strip(),
                notes=str(row["notes"]).strip(),
            )
            for row in reader
        ]


def parse_systems(raw_systems: str) -> list[str]:
    if raw_systems.strip().lower() == "all":
        return list(SYSTEMS)
    selected = [part.strip() for part in raw_systems.split(",") if part.strip()]
    unknown = [system for system in selected if system not in SYSTEMS]
    if unknown:
        raise ValueError(f"Unknown systems: {unknown}. Valid systems: all, {', '.join(SYSTEMS)}")
    return selected


def ensure_stage_docs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")
    (output_dir / "presentation_summary.md").write_text(build_presentation_summary(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_artifacts_for_system(system: str, state: RunState) -> bool:
    common_required = [DATABASE_PATH]
    faiss_required = [EMBEDDINGS_PATH, IMAGE_IDS_PATH, INDEX_PATH]

    missing = [path for path in common_required if not path.exists()]
    if system.startswith("faiss"):
        missing.extend(path for path in faiss_required if not path.exists())

    if missing:
        state.systems_skipped[system] = "Missing required artifact(s): " + ", ".join(
            str(path.relative_to(PROJECT_ROOT)) for path in missing
        )
        return False

    if system.startswith("qdrant"):
        return validate_qdrant_available(system, state)

    return True


def validate_qdrant_available(system: str, state: RunState) -> bool:
    settings = get_qdrant_settings()
    if settings.mode == "local" and not settings.path.exists():
        state.systems_skipped[system] = f"Local Qdrant path does not exist: {settings.path}"
        return False

    store_kwargs: dict[str, Any] = {"collection_name": settings.collection_name}
    if settings.mode == "server":
        store_kwargs["qdrant_url"] = settings.url
    else:
        store_kwargs["qdrant_path"] = settings.path

    store: QdrantStore | None = None
    try:
        store = QdrantStore(**store_kwargs)
        count = store.count()
        if count <= 0:
            state.systems_skipped[system] = f"Qdrant collection is empty: {settings.collection_name}"
            return False
    except Exception as exc:
        state.systems_skipped[system] = f"Qdrant unavailable: {exc}"
        return False
    finally:
        if store is not None:
            store.close()

    return True


def build_clip_encoder_if_needed(systems: list[str], state: RunState) -> CLIPEncoder | None:
    available_systems = [system for system in systems if system not in state.systems_skipped]
    if not available_systems:
        return None
    smoke_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.ml.clip_encoder import CLIPEncoder; CLIPEncoder(); print('clip ok')",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if smoke_result.returncode != 0:
        stderr = (smoke_result.stderr or "").strip()
        stdout = (smoke_result.stdout or "").strip()
        detail = stderr or stdout or f"process exited with code {smoke_result.returncode}"
        message = f"Could not initialize CLIP encoder in subprocess: {detail}"
        state.warnings.append(message)
        for system in available_systems:
            state.systems_skipped[system] = message
        return None
    try:
        return CLIPEncoder()
    except Exception as exc:
        message = f"Could not initialize CLIP encoder: {exc}"
        state.warnings.append(message)
        for system in available_systems:
            state.systems_skipped[system] = message
        return None


def build_qdrant_service(clip_encoder: CLIPEncoder) -> QdrantRetrievalService:
    settings = get_qdrant_settings()
    store_kwargs: dict[str, Any] = {"collection_name": settings.collection_name}
    if settings.mode == "server":
        store_kwargs["qdrant_url"] = settings.url
    else:
        store_kwargs["qdrant_path"] = settings.path
    return QdrantRetrievalService(
        clip_encoder=clip_encoder,
        qdrant_store=QdrantStore(**store_kwargs),
    )


def run_system(
    system: str,
    queries: list[ValidationQuery],
    *,
    top_k: int,
    candidate_pool_size: int,
    clip_encoder: CLIPEncoder,
    metadata_lookup: MetadataLookup,
    state: RunState,
) -> list[dict[str, Any]]:
    if system.startswith("faiss"):
        faiss = FaissValidationSearch(
            clip_encoder=clip_encoder,
            candidate_pool_size=candidate_pool_size,
        )
        runner = lambda query: run_faiss_query(  # noqa: E731
            system,
            query,
            faiss=faiss,
            top_k=top_k,
            metadata_lookup=metadata_lookup,
            state=state,
        )
        return collect_system_rows(system, queries, runner, state)

    service = build_qdrant_service(clip_encoder)
    try:
        runner = lambda query: run_qdrant_query(  # noqa: E731
            system,
            query,
            service=service,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            metadata_lookup=metadata_lookup,
            state=state,
        )
        return collect_system_rows(system, queries, runner, state)
    finally:
        service.close()


def collect_system_rows(
    system: str,
    queries: list[ValidationQuery],
    runner: Callable[[ValidationQuery], list[dict[str, Any]]],
    state: RunState,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        try:
            rows.extend(runner(query))
        except Exception as exc:
            state.warnings.append(f"{system} failed for {query.query_id} ({query.query}): {exc}")
    return rows


def run_faiss_query(
    system: str,
    query: ValidationQuery,
    *,
    faiss: FaissValidationSearch,
    top_k: int,
    metadata_lookup: MetadataLookup,
    state: RunState,
) -> list[dict[str, Any]]:
    if query.query_type in TEXT_QUERY_TYPES:
        results, note = faiss.search_text(
            query.query,
            top_k=top_k,
            rerank=system == "faiss_style_rerank",
        )
        return format_results(query, system, results, metadata_lookup, note)

    if query.query_type == "image_to_image":
        image_path = resolve_image_query_path(query, metadata_lookup, state)
        if image_path is None:
            return []
        results, note = faiss.search_image(
            image_path,
            top_k=top_k,
            rerank=system == "faiss_style_rerank",
        )
        if system == "faiss_baseline":
            note = "Image-to-image FAISS CLIP similarity search."
        return format_results(query, system, results, metadata_lookup, note)

    state.warnings.append(f"Unsupported query type for {query.query_id}: {query.query_type}")
    return []


def run_qdrant_query(
    system: str,
    query: ValidationQuery,
    *,
    service: QdrantRetrievalService,
    top_k: int,
    candidate_pool_size: int,
    metadata_lookup: MetadataLookup,
    state: RunState,
) -> list[dict[str, Any]]:
    if query.query_type == "image_to_image":
        image_path = resolve_image_query_path(query, metadata_lookup, state)
        if image_path is None:
            return []
        if system == "qdrant_object_rerank":
            results = service.search_by_image(image_path, top_k=top_k, rerank=False)
            return format_results(
                query,
                system,
                results,
                metadata_lookup,
                "No reliable object inferred for image-to-image query; used Qdrant image similarity.",
            )
        results = service.search_by_image(image_path, top_k=top_k, rerank=False)
        note = "Qdrant image similarity search."
        if system == "qdrant_filtered":
            note = "No reliable filter inferred for image-to-image query; used Qdrant image similarity."
        return format_results(query, system, results, metadata_lookup, note)

    if query.query_type not in TEXT_QUERY_TYPES:
        state.warnings.append(f"Unsupported query type for {query.query_id}: {query.query_type}")
        return []

    if system == "qdrant_semantic":
        results = service.search_by_text(text=query.query, top_k=top_k, rerank=False)
        return format_results(query, system, results, metadata_lookup, "Qdrant semantic vector search.")

    if system == "qdrant_filtered":
        filters, note = infer_qdrant_filters(query)
        if filters:
            results = service.search_by_text(
                text=query.query,
                top_k=top_k,
                rerank=False,
                candidate_pool_size=candidate_pool_size,
                **filters,
            )
            if not results:
                results = service.search_by_text(text=query.query, top_k=top_k, rerank=False)
                note = f"{note}; filter returned no results, used Qdrant semantic fallback."
        else:
            results = service.search_by_text(text=query.query, top_k=top_k, rerank=False)
        return format_results(query, system, results, metadata_lookup, note)

    if system == "qdrant_object_rerank":
        requested_object = infer_object(query.query)
        candidates = service.search_by_text(
            text=query.query,
            top_k=candidate_pool_size,
            rerank=False,
            candidate_pool_size=candidate_pool_size,
        )
        if requested_object:
            results = ObjectAwareReranker.object_heavy().rerank(
                candidates,
                requested_object=requested_object,
                requested_keyword=requested_object,
            )[:top_k]
            note = f"Qdrant semantic candidates reranked for object='{requested_object}'."
        else:
            results = candidates[:top_k]
            note = "No reliable object inferred; used Qdrant semantic order."
        return format_results(query, system, results, metadata_lookup, note)

    raise ValueError(f"Unsupported Qdrant system: {system}")


def format_results(
    query: ValidationQuery,
    system: str,
    results: list[dict[str, Any]],
    metadata_lookup: MetadataLookup,
    note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        image_id = str(result.get("image_id") or "")
        metadata = metadata_lookup.get_image(image_id) if image_id else None
        keywords = result.get("keywords") or (metadata or {}).get("keywords") or []
        objects = result.get("detected_objects") or (metadata or {}).get("detected_objects") or []

        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "type": query.query_type,
                "system": system,
                "rank": rank,
                "image_id": image_id,
                "score": format_number(result.get("score", result.get("final_score"))),
                "image_path": str(result.get("file_path") or (metadata or {}).get("file_path") or ""),
                "matched_keywords": join_values(keywords),
                "detected_objects": join_values(objects),
                "brightness": format_number(result.get("brightness", (metadata or {}).get("brightness"))),
                "contrast": format_number(result.get("contrast", (metadata or {}).get("contrast"))),
                "saturation": format_number(result.get("saturation", (metadata or {}).get("saturation"))),
                "warmth": format_number(result.get("warmth", (metadata or {}).get("warmth"))),
                "notes": note,
            }
        )
    return rows


def resolve_image_query_path(
    query: ValidationQuery,
    metadata_lookup: MetadataLookup,
    state: RunState,
) -> Path | None:
    image_id = extract_image_id(query)
    if image_id is None:
        marker = f"{query.query_id}: placeholder image_to_image query skipped"
        if marker not in state.skipped_image_queries:
            state.skipped_image_queries.append(marker)
        return None

    path = metadata_lookup.resolve_image_path(image_id)
    if path is None:
        marker = f"{query.query_id}: image_id '{image_id}' not found or file missing"
        if marker not in state.skipped_image_queries:
            state.skipped_image_queries.append(marker)
        return None
    return path


def extract_image_id(query: ValidationQuery) -> str | None:
    raw_query = query.query.strip()
    if raw_query and not raw_query.upper().startswith("REPLACE_WITH_REAL_IMAGE_ID"):
        return raw_query

    match = re.search(r"(?:image_id|id)\s*[:=]\s*([A-Za-z0-9_-]+)", query.notes)
    if match:
        return match.group(1)
    return None


def build_text_query_metadata(text: str) -> ImageMetadata | None:
    query_text = text.lower()
    brightness = None
    contrast = None
    saturation = None
    warmth = None

    if any(token in query_text for token in ("dark", "moody", "night", "low key", "low-key")):
        brightness = 0.20
        contrast = 0.70
    if any(token in query_text for token in ("bright", "sunny", "airy", "high key", "high-key")):
        brightness = 0.80
    if any(token in query_text for token in ("warm", "golden", "sunset", "summer")):
        warmth = 0.80
    if any(token in query_text for token in ("cold", "blue", "snowy", "winter", "icy")):
        warmth = 0.20
    if any(token in query_text for token in ("vibrant", "colorful", "saturated", "neon")):
        saturation = 0.80
    if any(token in query_text for token in ("muted", "minimal", "pastel", "foggy")):
        saturation = 0.30
    if any(token in query_text for token in ("cinematic", "dramatic", "high contrast")) and contrast is None:
        contrast = 0.65
    if "low contrast" in query_text:
        contrast = 0.30

    if all(value is None for value in (brightness, contrast, saturation, warmth)):
        return None

    return ImageMetadata(
        image_id="__query_text__",
        file_path=text,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        warmth=warmth,
        color_histogram=None,
    )


def infer_qdrant_filters(query: ValidationQuery) -> tuple[dict[str, Any], str]:
    filters: dict[str, Any] = {}
    notes: list[str] = []

    requested_object = infer_object(query.query)
    if query.query_type in {"object_text", "mixed_text"} and requested_object:
        filters["object_filter"] = requested_object
        notes.append(f"object_filter={requested_object}")

    style_filters = infer_style_filters(query.query)
    if query.query_type in {"style_text", "mixed_text"} and style_filters:
        filters.update(style_filters)
        notes.extend(f"{key}={value}" for key, value in style_filters.items())

    if not filters:
        return {}, "No reliable payload filter inferred; used Qdrant semantic fallback."
    return filters, "Applied Qdrant payload filter(s): " + ", ".join(notes)


def infer_object(query: str) -> str | None:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    for token in tokens:
        if token == "animal":
            return None
        if token in OBJECT_TERMS:
            return token
    return None


def infer_style_filters(query: str) -> dict[str, float]:
    text = query.lower()
    filters: dict[str, float] = {}
    if any(token in text for token in ("dark", "moody", "night", "low key", "low-key")):
        filters["max_brightness"] = 0.45
    if any(token in text for token in ("bright", "sunny", "airy")):
        filters["min_brightness"] = 0.55
    if any(token in text for token in ("warm", "golden", "sunset", "summer")):
        filters["min_warmth"] = 0.55
    if any(token in text for token in ("cold", "blue", "winter", "icy")):
        filters["max_warmth"] = 0.45
    if any(token in text for token in ("vibrant", "colorful", "saturated", "neon")):
        filters["min_saturation"] = 0.50
    if any(token in text for token in ("pastel", "muted", "minimal")):
        filters["max_saturation"] = 0.45
    if "high contrast" in text:
        filters["min_contrast"] = 0.55
    if "low contrast" in text:
        filters["max_contrast"] = 0.45
    return filters


def write_run_summary(
    output_dir: Path,
    *,
    queries: list[ValidationQuery],
    top_k: int,
    candidate_pool_size: int,
    state: RunState,
) -> None:
    lines = [
        "# Validation Results Run Summary",
        "",
        f"- Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- Queries loaded: {len(queries)}",
        f"- top_k: {top_k}",
        f"- candidate_pool_size: {candidate_pool_size}",
        f"- Systems attempted: {', '.join(state.systems_attempted) or 'none'}",
        f"- Systems completed: {', '.join(state.systems_completed) or 'none'}",
        f"- Systems skipped: {', '.join(state.systems_skipped) or 'none'}",
        "",
        "## Result Rows",
        "",
        "| System | Rows |",
        "| --- | ---: |",
    ]
    for system in SYSTEMS:
        lines.append(f"| `{system}` | {state.rows_per_system.get(system, 0)} |")

    lines.extend(["", "## Skipped Systems", ""])
    if state.systems_skipped:
        for system, reason in state.systems_skipped.items():
            lines.append(f"- `{system}`: {reason}")
    else:
        lines.append("- None")

    lines.extend(["", "## Skipped Image-to-Image Queries", ""])
    if state.skipped_image_queries:
        for item in state.skipped_image_queries:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings and Assumptions", ""])
    if state.warnings:
        for warning in state.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.append("- Image-to-image placeholder rows are skipped until real local image IDs are provided.")
    lines.append("- This stage generates ranked candidates only; relevance labels and metrics are not computed here.")

    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_readme() -> str:
    return """# Validation Results

This experiment stage generates ranked top-k search outputs for the validation queries defined in `experiments/08_validation_set/queries.csv`.

The outputs are intended for the next evaluation step: manual relevance labeling and later computation of Precision@5, Precision@10, nDCG@10, and MRR. This stage does not compute those metrics.

## Run

```powershell
python experiments/scripts/generate_validation_results.py
```

Optional arguments:

```powershell
python experiments/scripts/generate_validation_results.py --queries-path experiments/08_validation_set/queries.csv --output-dir experiments/09_validation_results --top-k 10 --candidate-pool-size 100 --systems all
```

`--systems` accepts `all` or a comma-separated list of `faiss_baseline`, `faiss_style_rerank`, `qdrant_semantic`, `qdrant_filtered`, and `qdrant_object_rerank`.

## Outputs

| File | Meaning |
| --- | --- |
| `results_faiss_baseline.csv` | Exact FAISS CLIP semantic baseline results. |
| `results_faiss_style_rerank.csv` | FAISS semantic candidates reranked with the existing style reranker when style cues are available. |
| `results_qdrant_semantic.csv` | Qdrant semantic vector search without payload filters. |
| `results_qdrant_filtered.csv` | Qdrant results using inferred object/style payload filters when reliable, otherwise semantic fallback. |
| `results_qdrant_object_rerank.csv` | Qdrant semantic candidates reranked with object-aware reranking when an object can be inferred. |
| `all_validation_results.csv` | Concatenated rows from all generated system result CSVs. |
| `run_summary.md` | Run timestamp, completed/skipped systems, row counts, warnings, and skipped image-to-image queries. |

All result CSVs use the same schema so they can be copied into the manual relevance-labeling workflow.
"""


def build_presentation_summary() -> str:
    return """# Validation Search Results

- Generated top-k candidate results for the validation query set.
- Stored separate CSV outputs for FAISS, Qdrant, filtering, and reranking systems.
- Preserved a combined CSV for manual relevance labeling.
- Logged skipped systems and placeholder image-to-image queries in `run_summary.md`.

| System | Purpose |
| --- | --- |
| FAISS baseline | Exact CLIP semantic baseline |
| FAISS + style rerank | Tests whether style-aware reranking improves visual style matching |
| Qdrant semantic | Production-like vector database search |
| Qdrant filtered | Tests payload filters for style/object/keyword control |
| Qdrant object rerank | Tests object-aware reranking over semantic candidates |

This stage connects the prepared validation set to the final evaluation pipeline by producing ranked candidates ready for manual relevance labels.
"""


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def join_values(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    try:
        return ";".join(str(value) for value in values if str(value).strip())
    except TypeError:
        return str(values)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be greater than 0")

    queries_path = args.queries_path if args.queries_path.is_absolute() else PROJECT_ROOT / args.queries_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    systems = parse_systems(args.systems)
    state = RunState(systems_attempted=list(systems))

    ensure_stage_docs(output_dir)
    queries = load_queries(queries_path)
    metadata_lookup = MetadataLookup(DATABASE_PATH)

    for system in systems:
        validate_artifacts_for_system(system, state)

    clip_encoder = build_clip_encoder_if_needed(systems, state)
    all_rows: list[dict[str, Any]] = []

    for system in systems:
        if system in state.systems_skipped or clip_encoder is None:
            rows: list[dict[str, Any]] = []
        else:
            rows = run_system(
                system,
                queries,
                top_k=args.top_k,
                candidate_pool_size=args.candidate_pool_size,
                clip_encoder=clip_encoder,
                metadata_lookup=metadata_lookup,
                state=state,
            )
            state.systems_completed.append(system)

        state.rows_per_system[system] = len(rows)
        write_csv(output_dir / SYSTEM_OUTPUTS[system], rows)
        all_rows.extend(rows)

    write_csv(output_dir / "all_validation_results.csv", all_rows)
    write_run_summary(
        output_dir,
        queries=queries,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        state=state,
    )

    print(f"Loaded queries: {len(queries)}")
    print(f"Completed systems: {', '.join(state.systems_completed) or 'none'}")
    print(f"Skipped systems: {', '.join(state.systems_skipped) or 'none'}")
    print(f"Saved outputs: {output_dir}")


if __name__ == "__main__":
    main()
