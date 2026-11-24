"""Small, report-only helpers shared by every analysis command.

One input surface (``add_input_args`` / ``load_input``), one cohort filter
(``--ids-file``), one point-date rule (``point_years``) and one report writer
(``save_reports``) so every analysis can be pinned to a named revision and
leaves the same manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

# Every module in analyses/ uses this import pattern: the installed package
# name first, the source tree only when the package is not installed (the
# ``python analyses/x.py`` route). Only the missing package itself selects the
# fallback, so a broken dependency is never masked.
try:
    from iwac_pipeline.processing._common import load_subset_dataframe, write_run_manifest
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "post-processing"))
    from _common import load_subset_dataframe, write_run_manifest  # noqa: E402
from iwac_common.paths import workspace_root  # noqa: E402

REPO_ROOT = workspace_root()
from iwac_common.field_mappers import parse_pub_date  # noqa: E402
from iwac_common.repos import PRIVATE_REPO_ID  # noqa: E402
from iwac_common.schema import validate_ids  # noqa: E402

MISSING = "[missing]"
METADATA = ("o:id", "iwac_url", "title", "country", "newspaper", "newspaper_ids",
            "pub_date", "language")
OUTPUT_DIR = REPO_ROOT / "analyses" / "output"
POINT_PRECISIONS = ("day", "month", "year")


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


def point_years(values) -> tuple[pd.Series, pd.Series]:
    """``(year, precision)`` under the one dating rule shared by all analyses.

    Only day, month and year dates carry a year. Ranges (``2000/2001``),
    unrecognised strings and missing dates are NA rather than assigned their
    first four characters or a range's start year, so an item belongs to the
    same year in every report or to none (docs/lda-model-bundles.md).
    ``precision`` is ``day|month|year|range|other`` or ``[missing]``.
    """
    values = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype=object)
    parsed = [parse_pub_date(string(value)) for value in values]
    precision = pd.Series([p[1] or MISSING for p in parsed], index=values.index, dtype=object)
    years = pd.Series([p[0] if p[1] in POINT_PRECISIONS else None for p in parsed],
                      index=values.index, dtype="Int64")
    return years, precision


def dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``point_years`` of a frame's ``pub_date`` column (absent column = missing)."""
    return point_years(series(df, "pub_date"))


def date_exclusions(precision: pd.Series) -> dict:
    """Counts of rows without a point year, by precision (for every report)."""
    excluded = precision[~precision.isin(POINT_PRECISIONS)]
    return {"rule": "point dates only (day/month/year); ranges and unparsed dates excluded",
            "excluded": int(len(excluded)),
            "by_precision": {str(k): int(v) for k, v in excluded.value_counts().sort_index().items()}}


def in_window(years: pd.Series, year_min=None, year_max=None) -> pd.Series:
    """Boolean mask of point years inside an optional inclusive window."""
    keep = years.notna()
    if year_min is not None:
        keep &= years.ge(year_min).fillna(False)
    if year_max is not None:
        keep &= years.le(year_max).fillna(False)
    return keep.fillna(False).astype(bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_link(row) -> str:
    """The row's ``iwac_url``, else the public item page built the same way the
    upload builds it (site slug and ``IWAC_PUBLIC_BASE_URL`` included)."""
    from iwac_common.omeka_client import item_page_url

    url = string(row.get("iwac_url"))
    return url or item_page_url(quote(string(row["o:id"]), safe=""))


def metadata(row, *, excerpt_chars=0, text_column="OCR") -> dict:
    result = {column: string(row.get(column)) for column in METADATA}
    result["iwac_url"] = source_link(row)
    if excerpt_chars:
        result["text_excerpt"] = string(row.get(text_column))[:excerpt_chars]
    return result


def add_input_args(parser: argparse.ArgumentParser, *, excerpts=False, source_default="local",
                   output_dir: Path | None = None, cohort=False, companion_index=False,
                   seed=False) -> None:
    """The shared input/provenance surface of every analysis command.

    ``--source csv`` is accepted as the historical spelling of ``local``.
    ``cohort`` adds ``--ids-file``/``--ids-where``; ``companion_index`` adds
    ``--index-input`` for analyses that also read the authority file.
    """
    parser.add_argument("--input", type=Path, help="Offline CSV/Parquet; fingerprint recorded in manifest")
    # None until resolved: an explicit --source hub conflicts with --input, a default does not.
    parser.add_argument("--source", choices=("local", "hub", "csv"), default=None,
                        help="local = verified revision-pinned mirror ('csv' is its historical alias); "
                             f"hub = read-only Hub load with existing credentials (default: {source_default})")
    parser.set_defaults(source_default=source_default)
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", default="articles")
    parser.add_argument("--revision", help="Pin Hub/mirror revision; provenance declaration for --input")
    parser.add_argument("--output-dir", type=Path, default=output_dir or OUTPUT_DIR)
    if companion_index:
        parser.add_argument("--index-input", type=Path,
                            help="Offline index CSV/Parquet; required with --input")
    if cohort:
        parser.add_argument("--ids-file", type=Path,
                            help="Restrict the analysis to these o:id values: a CSV with an o:id "
                                 "column, or a text file with one ID per line")
        parser.add_argument("--ids-where", metavar="COLUMN",
                            help="With a CSV --ids-file, keep only rows whose COLUMN is true, e.g. "
                                 "stable_outlet_cohort in corpus_coverage_inclusion.csv")
    if seed:
        parser.add_argument("--seed", type=int, default=42, help="Seed for every resampling step")
    if excerpts:
        parser.add_argument("--excerpt-chars", type=int, default=0,
                            help="Explicitly include up to N text characters; default metadata only")
        parser.add_argument("--text-column", default="OCR")


def _normalized_source(args) -> str:
    """Resolve ``--source``: ``csv`` -> ``local``; unset -> local with --input, else the script default."""
    source = args.source
    if source is None:
        source = "local" if getattr(args, "input", None) else getattr(args, "source_default", "local")
    elif source == "csv":
        source = "local"
    args.source = source
    return source


def _read_token(source: str):
    # Read-only access never starts an interactive login or writes datasets.
    if source != "hub":
        return None
    from huggingface_hub import get_token
    return get_token()


def load_input(args, *, columns=None, loader=None) -> pd.DataFrame:
    """Load the analysed subset with its provenance in ``df.attrs``.

    ``loader`` defaults to ``load_subset_dataframe``; a script passes its own
    module-level reference so it stays the single patch point in tests.
    """
    loader = loader or load_subset_dataframe
    source = _normalized_source(args)
    if getattr(args, "excerpt_chars", 0) < 0:
        raise ValueError("--excerpt-chars must be non-negative")
    if args.input:
        if source == "hub":
            raise ValueError("--input is an offline source; do not combine with --source hub")
        if args.input.suffix.lower() not in (".csv", ".parquet"):
            raise ValueError("--input must be a .csv or .parquet file")
        df = loader(args.repo, args.config, source="local", csv_path=args.input, columns=columns)
        df.attrs["iwac_source_revision"] = args.revision
        df.attrs["input_sha256"] = sha256(args.input)
        df.attrs["revision_status"] = "user_declared" if args.revision else "unknown"
    else:
        df = loader(args.repo, args.config, source=source, token=_read_token(source),
                    revision=args.revision, columns=columns)
        df.attrs["revision_status"] = "pinned"
    if "o:id" in df:
        # The historical loader casts IDs to str, including null sentinels.
        df["o:id"] = df["o:id"].astype(str).str.strip()
        if df["o:id"].str.lower().isin({"nan", "none", "<na>", "nat", "null"}).any():
            raise ValueError("Research input contains missing source IDs")
    validate_ids(df, label="research input")
    return df


def load_companion(args, df, config: str, *, columns=None, loader=None,
                   path: Path | None = None) -> pd.DataFrame:
    """Load a second subset (e.g. ``index``) at the analysed subset's revision.

    An offline ``--input`` needs an equally explicit companion file: mixing an
    offline snapshot with whatever mirror happens to be present would silently
    combine revisions.
    """
    loader = loader or load_subset_dataframe
    source = _normalized_source(args)
    if path:
        if path.suffix.lower() not in (".csv", ".parquet"):
            raise ValueError(f"{config} input must be a .csv or .parquet file")
        frame = loader(args.repo, config, source="local", csv_path=path, columns=columns)
        frame.attrs["input_sha256"] = sha256(path)
        return frame
    if args.input:
        raise ValueError(f"--input requires an explicit offline {config} file (--{config}-input)")
    return loader(args.repo, config, source=source, token=_read_token(source),
                  revision=df.attrs.get("iwac_source_revision"), columns=columns)


TRUE_LABELS = frozenset({"true", "1", "yes", "y", "t"})


def read_ids_file(path: Path, where: str | None = None) -> set[str]:
    """Source IDs from a CSV (``o:id`` column) or a one-ID-per-line text file."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "o:id" not in frame:
            raise ValueError(f"{path} has no o:id column")
        if where:
            if where not in frame:
                raise ValueError(f"{path} has no {where} column for --ids-where")
            frame = frame[frame[where].str.strip().str.casefold().isin(TRUE_LABELS)]
        ids = frame["o:id"].str.strip()
    else:
        if where:
            raise ValueError("--ids-where needs a CSV --ids-file")
        lines = path.read_text(encoding="utf-8").splitlines()
        ids = pd.Series([line.strip() for line in lines if line.strip() and not line.startswith("#")],
                        dtype=str)
    ids = set(ids[ids.ne("")])
    if not ids:
        raise ValueError(f"{path} selects no source IDs")
    return ids


def apply_ids_filter(df: pd.DataFrame, args) -> pd.DataFrame:
    """Restrict to an exported cohort (e.g. the stable-outlet ledger).

    Applied after loading so the cohort is defined on the same revision. IDs
    absent from the input mean the cohort came from another snapshot: that
    fails rather than silently shrinking the cohort. Counts travel in
    ``df.attrs["ids_filter"]`` into the summary and manifest.
    """
    path = getattr(args, "ids_file", None)
    if not path:
        if getattr(args, "ids_where", None):
            raise ValueError("--ids-where requires --ids-file")
        df.attrs["ids_filter"] = None
        return df
    ids = read_ids_file(path, getattr(args, "ids_where", None))
    present_ids = df["o:id"].astype(str)
    absent = ids - set(present_ids)
    if absent:
        raise ValueError(f"{len(absent)} cohort IDs are absent from this input (e.g. "
                         f"{sorted(absent)[:5]}); export the cohort from the same revision")
    keep = present_ids.isin(ids)
    out = df.loc[keep].copy()
    out.attrs = dict(df.attrs)
    out.attrs["ids_filter"] = {"file": str(path), "sha256": sha256(Path(path)),
                               "where": getattr(args, "ids_where", None), "ids_in_file": len(ids),
                               "rows_loaded": int(len(df)), "rows_kept": int(keep.sum()),
                               "rows_dropped": int((~keep).sum())}
    print(f"→ Cohort filter {path}: kept {int(keep.sum()):,} of {len(df):,} rows")
    return out


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def save_reports(args, df, script: str, tables: dict[str, pd.DataFrame], summary: dict, *,
                 inputs: dict | None = None, writer=None) -> list[Path]:
    """Write CSVs, ``<script>_summary.json`` and the run manifest.

    A table key ending in ``.csv`` is used verbatim as the filename (keeps the
    historical names of the older reports); any other key becomes
    ``<script>_<key>.csv``. Non-finite numbers become JSON nulls.
    """
    writer = writer or write_run_manifest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision = df.attrs.get("iwac_source_revision")
    outputs = []
    for name, frame in tables.items():
        frame = frame.copy()
        # Source identity travels with exported review rows, not only the manifest.
        frame["source_repository"] = args.repo
        frame["source_revision"] = revision or "unknown"
        path = args.output_dir / (name if name.endswith(".csv") else f"{script}_{name}.csv")
        frame.to_csv(path, index=False, encoding="utf-8")
        outputs.append(path)
    summary = {**summary,
               "source": {"repository": args.repo, "revision": revision or "unknown",
                          "revision_status": df.attrs.get("revision_status")},
               "ids_filter": df.attrs.get("ids_filter")}
    path = args.output_dir / f"{script}_summary.json"
    path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, default=str,
                               allow_nan=False) + "\n", encoding="utf-8")
    outputs.append(path)
    record = {"file": str(args.input) if getattr(args, "input", None) else None,
              "sha256": df.attrs.get("input_sha256"),
              "revision_status": df.attrs.get("revision_status"),
              "text_excerpts_enabled": bool(getattr(args, "excerpt_chars", 0)),
              "ids_filter": df.attrs.get("ids_filter")}
    # Optional sidecar inputs also belong to the reproducibility record.
    for key in ("difficult_ids", "evaluated_pairs", "index_input"):
        extra = getattr(args, key, None)
        if extra:
            record[key] = {"file": str(extra), "sha256": sha256(Path(extra))}
    record.update(inputs or {})
    writer(args.output_dir, script=script, repo_id=args.repo, revision=revision, args=args,
           outputs=outputs, inputs=record)
    print(f"✓ Saved {script} reports to {args.output_dir}")
    return outputs


def stable_rank(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{item_id}".encode()).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def shingles(text: str, size=3) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


class ShingleJaccard:
    """Word-shingle Jaccard computed lazily, only for the pairs actually scored.

    Shingle sets are built on first use and cached per ID, so memory grows with
    the candidate pairs, not with the corpus. ``texts`` maps ID -> raw text (a
    dict, or a pandas Series indexed by ID).
    """

    def __init__(self, texts, size: int = 3):
        self.texts, self.size, self.cache = texts, size, {}

    def grams(self, item_id: str) -> set:
        if item_id not in self.cache:
            self.cache[item_id] = shingles(string(self.texts.get(item_id)), self.size)
        return self.cache[item_id]

    def __call__(self, a: str, b: str) -> float:
        left, right = self.grams(a), self.grams(b)
        return len(left & right) / len(left | right) if left and right else float("nan")
