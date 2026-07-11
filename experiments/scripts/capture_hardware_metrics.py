from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "13_index_selection" / "hardware_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture host, Python, GPU, Qdrant and Docker metrics for a labeled run.")
    parser.add_argument("--label", required=True, help="Measurement label, e.g. 25k or 500k.")
    parser.add_argument("--container", default="photographer-style-qdrant")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", ""), help="Collection whose point count should be recorded.")
    parser.add_argument("--qdrant-path", type=Path, default=PROJECT_ROOT / "data" / "qdrant_storage")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024 * 1024)


def docker_stats(container: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
        row = json.loads(result.stdout.strip().splitlines()[0])
        row["available"] = True
        return row
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, json.JSONDecodeError) as exc:
        return {"available": False, "error": str(exc)}


def host_memory() -> dict[str, Any]:
    try:
        import psutil  # type: ignore
        memory = psutil.virtual_memory()
        return {"total_mb": memory.total / 1024**2, "available_mb": memory.available / 1024**2}
    except ImportError:
        return {"available": None, "note": "Install psutil for host RAM values."}


def gpu_info() -> dict[str, Any]:
    try:
        import torch
        result: dict[str, Any] = {"cuda_available": bool(torch.cuda.is_available()), "torch_cuda": torch.version.cuda}
        if torch.cuda.is_available():
            result["device_count"] = torch.cuda.device_count()
            result["devices"] = []
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                result["devices"].append({
                    "index": index,
                    "name": properties.name,
                    "total_memory_mb": properties.total_memory / 1024**2,
                })
        return result
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def qdrant_info(url: str, collection: str) -> dict[str, Any]:
    try:
        import requests
        response = requests.get(url.rstrip("/") + "/", timeout=5)
        payload = response.json()
        result = {"available": response.ok, "http_status": response.status_code, "root": payload}
        if collection:
            collection_response = requests.get(url.rstrip("/") + "/collections/" + collection, timeout=5)
            collection_payload = collection_response.json()
            result["collection"] = collection
            result["collection_http_status"] = collection_response.status_code
            result["collection_info"] = collection_payload
        return result
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"snapshots": []}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("snapshots"), list):
            return loaded
    except json.JSONDecodeError:
        pass
    return {"snapshots": []}


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    snapshots = []
    for sample_index in range(args.sample_count):
        snapshots.append({
            "sample_index": sample_index + 1,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "docker": docker_stats(args.container),
        })
        if sample_index + 1 < args.sample_count:
            time.sleep(max(0.0, args.sample_interval))

    output = resolve(args.output)
    document = load_existing(output)
    document["schema_version"] = 1
    document["snapshots"] = [snapshot for snapshot in document["snapshots"] if snapshot.get("label") != args.label]
    document["snapshots"].append({
        "label": args.label,
        "host": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory": host_memory(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "qdrant_client": package_version("qdrant-client"),
        },
        "gpu": gpu_info(),
        "qdrant": qdrant_info(args.qdrant_url, args.collection),
        "qdrant_storage_disk_mb": directory_size_mb(resolve(args.qdrant_path)),
        "container": args.container,
        "docker_samples": snapshots,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Captured hardware snapshot: {args.label}")
    print(f"Wrote: {output}")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
