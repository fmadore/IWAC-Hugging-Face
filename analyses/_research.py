"""Small, report-only helpers shared by the research validation commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from iwac_pipeline.processing._common import load_subset_dataframe, write_run_manifest
from iwac_common.paths import workspace_root

REPO_ROOT = workspace_root()
from iwac_common.field_mappers import parse_pub_date  # noqa: E402
from iwac_common.repos import PRIVATE_REPO_ID  # noqa: E402
from iwac_common.schema import validate_ids  # noqa: E402

MISSING = "[missing]"
METADATA = ("o:id", "iwac_url", "title", "country", "newspaper", "newspaper_ids",
            "pub_date", "language")


def present(value) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) > 0
    if value is None or pd.isna(value):
        return False
    return bool(str(value).strip())


def string(value, default="") -> str:
    return str(value).strip() if present(value) else default


def series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column] if column in df else pd.Series(None, index=df.index, dtype=object)


def dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    parsed = [parse_pub_date(string(value)) for value in series(df, "pub_date")]
    return (pd.Series([p[0] for p in parsed], index=df.index, dtype="Int64"),
            pd.Series([p[1] or MISSING for p in parsed], index=df.index, dtype=object))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_link(row) -> str:
    url = string(row.get("iwac_url"))
    return url or "https://islam.zmo.de/s/westafrica/item/" + quote(string(row["o:id"]), safe="")


def metadata(row, *, excerpt_chars=0, text_column="OCR") -> dict:
    result = {column: string(row.get(column)) for column in METADATA}
    result["iwac_url"] = source_link(row)
    if excerpt_chars:
        result["text_excerpt"] = string(row.get(text_column))[:excerpt_chars]
    return result


def add_input_args(parser: argparse.ArgumentParser, *, excerpts=False) -> None:
    parser.add_argument("--input", type=Path, help="Offline CSV/Parquet; fingerprint recorded in manifest")
    parser.add_argument("--source", choices=("local", "hub"), default="local")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", default="articles")
    parser.add_argument("--revision", help="Pin Hub/mirror revision; provenance declaration for --input")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analyses" / "output")
    if excerpts:
        parser.add_argument("--excerpt-chars", type=int, default=0,
                            help="Explicitly include up to N text characters; default metadata only")
        parser.add_argument("--text-column", default="OCR")


def load_input(args, *, columns=None) -> pd.DataFrame:
    if getattr(args, "excerpt_chars", 0) < 0:
        raise ValueError("--excerpt-chars must be non-negative")
    if args.input:
        if args.source == "hub":
            raise ValueError("--input is an offline source; do not combine with --source hub")
        if args.input.suffix.lower() not in (".csv", ".parquet"):
            raise ValueError("--input must be a .csv or .parquet file")
        df = load_subset_dataframe(args.repo, args.config, source="local",
                                   csv_path=args.input, columns=columns)
        df.attrs["iwac_source_revision"] = args.revision
        df.attrs["input_sha256"] = sha256(args.input)
        df.attrs["revision_status"] = "user_declared" if args.revision else "unknown"
    else:
        # Read-only access never starts an interactive login or writes datasets.
        from huggingface_hub import get_token
        df = load_subset_dataframe(args.repo, args.config, source=args.source,
                                   token=get_token() if args.source == "hub" else None,
                                   revision=args.revision, columns=columns)
        df.attrs["revision_status"] = "pinned"
    if "o:id" in df:
        # The historical loader casts IDs to str, including null sentinels.
        df["o:id"] = df["o:id"].astype(str).str.strip()
        if df["o:id"].str.lower().isin({"nan", "none", "<na>", "nat", "null"}).any():
            raise ValueError("Research input contains missing source IDs")
    validate_ids(df, label="research input")
    return df


def save_reports(args, df, script: str, tables: dict[str, pd.DataFrame], summary: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in tables.items():
        frame = frame.copy()
        # Source identity travels with exported review rows, not only the manifest.
        frame["source_repository"] = args.repo
        frame["source_revision"] = df.attrs.get("iwac_source_revision") or "unknown"
        path = args.output_dir / f"{script}_{name}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)
    path = args.output_dir / f"{script}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    outputs.append(path)
    inputs = {"file": str(args.input) if args.input else None,
              "sha256": df.attrs.get("input_sha256"),
              "revision_status": df.attrs.get("revision_status"),
              "text_excerpts_enabled": bool(getattr(args, "excerpt_chars", 0))}
    # Optional sidecar inputs also belong to the reproducibility record.
    for key in ("difficult_ids", "evaluated_pairs"):
        extra = getattr(args, key, None)
        if extra:
            inputs[key] = {"file": str(extra), "sha256": sha256(extra)}
    write_run_manifest(args.output_dir, script=script, repo_id=args.repo,
                       revision=df.attrs.get("iwac_source_revision"), args=args,
                       outputs=outputs, inputs=inputs)
    print(f"Saved {script} reports to {args.output_dir}")


def stable_rank(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{item_id}".encode()).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
