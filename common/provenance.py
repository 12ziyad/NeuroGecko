"""Load the JSON subset of YAML used by the auditable parameter registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "config" / "proxies.yaml"
REQUIRED_FIELDS = {
    "value", "units", "species", "sample_size", "source", "confidence",
    "recording_temp_C", "notes",
}


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate metadata without requiring a general YAML parser.

    proxies.yaml intentionally uses JSON syntax, which is valid YAML 1.2.
    Do not silently accept a malformed registry or missing provenance fields.
    """
    selected = Path(path) if path is not None else DEFAULT_REGISTRY
    registry = json.loads(selected.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported provenance registry schema")
    entries = registry.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Registry must contain a nonempty entries mapping")
    for name, entry in entries.items():
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"{name}: missing provenance fields {sorted(missing)}")
        if entry["confidence"] not in {"verified", "likely", "uncertain"}:
            raise ValueError(f"{name}: invalid confidence")
        if not entry["source"] or not entry["notes"]:
            raise ValueError(f"{name}: source and notes must be nonempty")
    return registry


def parameter_value(name: str, path: str | Path | None = None) -> Any:
    """Return one value, raising KeyError rather than inventing a fallback."""
    return load_registry(path)["entries"][name]["value"]
