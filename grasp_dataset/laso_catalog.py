"""Build and query the LASO object-ID to semantic-class catalog."""

from __future__ import annotations

import json
from pathlib import Path


def build(data_root: Path) -> dict[str, list[str]]:
    """Collect category -> object IDs from exported LASO metadata."""
    categories: dict[str, set[str]] = {}
    seen: dict[str, str] = {}
    for metadata_file in sorted(data_root.glob("**/metadata.json")):
        metadata = json.loads(metadata_file.read_text())
        if not str(metadata.get("schema", "")).startswith("laso-"):
            continue
        object_id = str(metadata.get("object_id") or metadata_file.parent.name)
        category = metadata.get("class")
        if not category or not isinstance(category, str):
            raise ValueError(f"{metadata_file} has no string 'class'")
        if object_id in seen and seen[object_id] != category:
            raise ValueError(f"LASO object {object_id} has conflicting classes: {seen[object_id]} and {category}")
        seen[object_id] = category
        categories.setdefault(category, set()).add(object_id)
    if not categories:
        raise ValueError(f"No LASO metadata.json files found under {data_root}")
    return {category: sorted(ids) for category, ids in sorted(categories.items())}


def load(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or any(
        not isinstance(category, str) or not isinstance(ids, list)
        or any(not isinstance(object_id, str) for object_id in ids)
        for category, ids in payload.items()
    ):
        raise ValueError(f"{path} must contain category -> list[object ID]")
    return payload


def object_type(object_id: str, catalog: dict[str, list[str]]) -> str | None:
    matches = [category for category, ids in catalog.items() if object_id in ids]
    if len(matches) > 1:
        raise ValueError(f"LASO object {object_id} appears in multiple categories: {matches}")
    return matches[0] if matches else None

