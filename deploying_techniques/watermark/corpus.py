from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_watermark_corpus(path: str | Path, expected_method: str | None = None) -> list[dict[str, Any]]:
    """Load and validate a committed smoke corpus without network access."""

    corpus_path = Path(path)
    records = []
    with corpus_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {
                "id",
                "method",
                "source_index",
                "prompt",
                "natural_text",
                "preparation",
                "content_sha256",
            } - record.keys()
            if missing:
                raise ValueError(f"{corpus_path}:{line_number} missing fields: {sorted(missing)}")
            if expected_method is not None and record["method"] != expected_method:
                raise ValueError(
                    f"{corpus_path}:{line_number} has method={record['method']!r}; "
                    f"expected {expected_method!r}"
                )
            digest = hashlib.sha256(
                (record["prompt"] + "\0" + record["natural_text"]).encode("utf-8")
            ).hexdigest()
            if digest != record["content_sha256"]:
                raise ValueError(f"{corpus_path}:{line_number} failed content checksum")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {corpus_path}")
    return records
