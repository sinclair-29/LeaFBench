#!/usr/bin/env python3
"""Create the small, offline C4 prompt corpora used by watermark smoke tests.

This is intentionally a one-off data preparation utility. Runtime smoke tests
read the committed JSONL files and never import ``datasets`` or access the
network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable


SOURCE_DATASET = "allenai/c4"
SOURCE_CONFIG = "realnewslike"
SOURCE_SPLIT = "train"
SOURCE_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
TOKENIZER_NAME = "facebook/opt-1.3b"
TOKENIZER_REVISION = "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"
DEFAULT_SEED = 1234
DEFAULT_SHUFFLE_BUFFER_SIZE = 10_000
DEFAULT_NUM_RECORDS = 5
COMPLETION_LENGTH = 200
DATASETS_VERSION = "4.0.0"
TRANSFORMERS_VERSION = "4.54.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/watermark"),
        help="Directory receiving the method-specific watermark smoke corpora.",
    )
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_SIZE,
        help="Number of initial streaming rows shuffled before filtering.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=TOKENIZER_NAME,
        help="Tokenizer ID or local OPT tokenizer path used for paper token counts.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing JSONL files.")
    return parser.parse_args()


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def common_preparation(
    *,
    rule: str,
    seed: int,
    shuffle_buffer_size: int,
    selection_rank: int,
    source_token_count: int,
    prompt_token_count: int,
    natural_text_token_count: int,
) -> dict[str, Any]:
    return {
        "rule": rule,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "add_special_tokens": False,
        "seed": seed,
        "selection": "shuffle_first_streaming_buffer_then_take_first_eligible",
        "shuffle_buffer_size": shuffle_buffer_size,
        "selection_rank": selection_rank,
        "source_token_count": source_token_count,
        "prompt_token_count": prompt_token_count,
        "natural_text_token_count": natural_text_token_count,
    }


def prepare_kgw(tokenizer: Any, token_ids: list[int]) -> tuple[str, str, dict[str, Any]] | None:
    # The KGW experiments trim a fixed 200-token suffix and require the
    # remaining prompt to contain at least 50 tokens.
    if len(token_ids) < 50 + COMPLETION_LENGTH:
        return None
    prompt_ids = token_ids[:-COMPLETION_LENGTH]
    natural_ids = token_ids[-COMPLETION_LENGTH:]
    return (
        decode(tokenizer, prompt_ids),
        decode(tokenizer, natural_ids),
        {
            "completion_length": COMPLETION_LENGTH,
            "minimum_prompt_tokens": 50,
            "natural_text_slice": "final_200_tokens",
        },
    )


def prepare_opt(tokenizer: Any, token_ids: list[int]) -> tuple[str, str, dict[str, Any]] | None:
    # Appendix B.1: ignore texts below 250 tokens. For texts through 400
    # tokens, remove the final 200 and use the remainder as the prompt. For
    # longer texts, use the first 200 tokens and discard the rest for model
    # input. We retain that discarded remainder as natural_text for auditing.
    source_length = len(token_ids)
    if source_length < 250:
        return None
    if source_length <= 400:
        prompt_ids = token_ids[:-COMPLETION_LENGTH]
        natural_ids = token_ids[-COMPLETION_LENGTH:]
        branch = "250_to_400"
        natural_slice = "final_200_tokens_removed_by_paper_rule"
    else:
        prompt_ids = token_ids[:COMPLETION_LENGTH]
        natural_ids = token_ids[COMPLETION_LENGTH:]
        branch = "over_400"
        natural_slice = "remainder_discarded_by_paper_rule_retained_for_audit"
    return (
        decode(tokenizer, prompt_ids),
        decode(tokenizer, natural_ids),
        {
            "minimum_source_tokens": 250,
            "long_text_boundary_tokens": 400,
            "maximum_prompt_tokens": 200,
            "branch": branch,
            "natural_text_slice": natural_slice,
            "natural_text_used_by_paper": False,
        },
    )


def prepare_morphmark(tokenizer: Any, token_ids: list[int]) -> tuple[str, str, dict[str, Any]] | None:
    # MorphMark follows MarkLLM: first 30 tokens are the prompt. Retain the
    # following 200-token natural continuation, matching its generation-length
    # evaluation window, so the smoke artifact stays compact and comparable.
    prompt_length = 30
    natural_length = 200
    if len(token_ids) < prompt_length + natural_length:
        return None
    prompt_ids = token_ids[:prompt_length]
    natural_ids = token_ids[prompt_length : prompt_length + natural_length]
    return (
        decode(tokenizer, prompt_ids),
        decode(tokenizer, natural_ids),
        {
            "prompt_prefix_tokens": prompt_length,
            "natural_text_window_tokens": natural_length,
            "natural_text_slice": "tokens_30_through_229",
        },
    )


PREPARERS: dict[
    str,
    tuple[str, str, Callable[[Any, list[int]], tuple[str, str, dict[str, Any]] | None]],
] = {
    "kgw": ("kgw_smoke.jsonl", "completion_length", prepare_kgw),
    "opt": ("opt_smoke.jsonl", "opt_250_400", prepare_opt),
    "morphmark": ("morphmark_smoke.jsonl", "first_30_tokens", prepare_morphmark),
    "watermod": ("watermod_smoke.jsonl", "completion_length", prepare_kgw),
}


def build_record(
    *,
    method: str,
    source_index: int,
    row: dict[str, Any],
    prompt: str,
    natural_text: str,
    preparation: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "id": f"{method}_{preparation['selection_rank']:04d}",
        "method": method,
        "source": {
            "dataset": SOURCE_DATASET,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "revision": SOURCE_REVISION,
        },
        "source_index": source_index,
        "prompt": prompt,
        "natural_text": natural_text,
        "preparation": preparation,
    }
    if row.get("url"):
        record["source"]["url"] = row["url"]
    if row.get("timestamp"):
        record["source"]["timestamp"] = row["timestamp"]
    record["content_sha256"] = hashlib.sha256(
        (prompt + "\0" + natural_text).encode("utf-8")
    ).hexdigest()
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force to replace it.")
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_records < 1:
        raise ValueError("--num-records must be positive")
    if args.shuffle_buffer_size < args.num_records:
        raise ValueError("--shuffle-buffer-size must be at least --num-records")

    try:
        import datasets
        import transformers
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Data preparation requires the repository dependencies `datasets` and `transformers`."
        ) from exc
    if datasets.__version__ != DATASETS_VERSION or transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(
            "Reproducible preparation requires "
            f"datasets=={DATASETS_VERSION} and transformers=={TRANSFORMERS_VERSION}; "
            f"found datasets=={datasets.__version__}, transformers=={transformers.__version__}."
        )

    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    if args.tokenizer_path == TOKENIZER_NAME:
        tokenizer_kwargs["revision"] = TOKENIZER_REVISION
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, **tokenizer_kwargs)

    dataset = load_dataset(
        SOURCE_DATASET,
        SOURCE_CONFIG,
        split=SOURCE_SPLIT,
        streaming=True,
        revision=SOURCE_REVISION,
    )
    candidates: list[tuple[int, dict[str, Any]]] = []
    for source_index, row in enumerate(dataset):
        candidates.append((source_index, row))
        if len(candidates) == args.shuffle_buffer_size:
            break
    if len(candidates) != args.shuffle_buffer_size:
        raise RuntimeError(
            f"C4 stream ended after {len(candidates)} rows; expected {args.shuffle_buffer_size}."
        )
    random.Random(args.seed).shuffle(candidates)

    collected: dict[str, list[dict[str, Any]]] = {method: [] for method in PREPARERS}
    for source_index, row in candidates:
        token_ids = tokenizer.encode(row["text"], add_special_tokens=False)
        for method, (_, rule, prepare) in PREPARERS.items():
            records = collected[method]
            if len(records) >= args.num_records:
                continue
            prepared = prepare(tokenizer, token_ids)
            if prepared is None:
                continue
            prompt, natural_text, rule_metadata = prepared
            preparation = common_preparation(
                rule=rule,
                seed=args.seed,
                shuffle_buffer_size=args.shuffle_buffer_size,
                selection_rank=len(records) + 1,
                source_token_count=len(token_ids),
                prompt_token_count=len(tokenizer.encode(prompt, add_special_tokens=False)),
                natural_text_token_count=len(
                    tokenizer.encode(natural_text, add_special_tokens=False)
                ),
            )
            preparation.update(rule_metadata)
            preparation["datasets_version"] = DATASETS_VERSION
            preparation["transformers_version"] = TRANSFORMERS_VERSION
            records.append(
                build_record(
                    method=method,
                    source_index=source_index,
                    row=row,
                    prompt=prompt,
                    natural_text=natural_text,
                    preparation=preparation,
                )
            )
        if all(len(records) == args.num_records for records in collected.values()):
            break

    incomplete = {
        method: len(records)
        for method, records in collected.items()
        if len(records) != args.num_records
    }
    if incomplete:
        raise RuntimeError(f"Not enough eligible records in shuffle buffer: {incomplete}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method, (filename, _, _) in PREPARERS.items():
        path = args.output_dir / filename
        write_jsonl(path, collected[method], args.force)
        print(f"Wrote {len(collected[method])} {method} records to {path}")


if __name__ == "__main__":
    main()
