"""Verify the complete card/Parquet schema at one immutable Hub revision.

The write gateway (``hub.push_datasets_verified``) serialises every Parquet
shard *and* the card's ``dataset_info``/``configs`` entries for the written
configs, then lands them together in one ``create_commit`` with a parent-SHA
precondition. The card's declared ``features`` are generated from the same
Arrow schema as the shards (:func:`_features_yaml`), so the two cannot drift in
a successful commit.

This module is the post-commit read-only check of that result, pinned to the
new revision: the published Parquet footers must equal the prepared schema and
columns, and the card must declare exactly that schema. A mismatch raises
:class:`CardSchemaError`, which the gateway reports as "commit landed but
verification failed" so the revision is inspected before any retry.

**Why it matters.** A card whose ``features`` disagree with the Parquet makes
``load_dataset`` raise ``CastError: Couldn't cast … because column names don't
match`` — the subset becomes unloadable for every consumer and for this
pipeline's own next run, since ``hub_merge`` loads the Hub copy before merging.
That happened on 2026-08-06, when writes still went through ``push_to_hub``
(which refreshed the byte sizes but not the feature list); the atomic commit
replaced that flow.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pyarrow as pa
from huggingface_hub import DatasetCard, hf_hub_download
from rich.console import Console

from .hub import get_repo_revision, iter_parquet_shards, resolve_hf_token


class CardSchemaError(RuntimeError):
    """The card's declared schema, the published parquet and the prepared
    dataset disagree. A subset in that state is unloadable until fixed."""


def _narrow(dtype: pa.DataType) -> pa.DataType:
    """``large_string`` → ``string``, recursively through lists.

    The two are the same logical type, ``string`` is what every hand-written
    entry in these cards uses, and it matches the ``num_bytes`` ``push_to_hub``
    records (which assumes 4-byte offsets). Normalising *both* sides before
    comparing is also what stops this guard from failing on every push.
    """
    if pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.list_(_narrow(dtype.value_type))
    if pa.types.is_fixed_size_list(dtype):
        return pa.list_(_narrow(dtype.value_type), dtype.list_size)
    if pa.types.is_struct(dtype):
        return pa.struct([pa.field(f.name, _narrow(f.type)) for f in dtype])
    return dtype


def _parquet_schema(repo: str, config_name: str, token: Optional[str], revision: Optional[str] = None) -> pa.Schema:
    """Arrow schema of the config's parquet on the Hub, read from the footer only
    (no shard download). All shards of a config share one schema."""
    schema = None
    for _, parquet in iter_parquet_shards(
        repo, config_name, revision=revision, token=token, error=CardSchemaError
    ):
        current = _normalize_schema(parquet.schema_arrow)
        if schema is not None and not schema.equals(current):
            raise CardSchemaError(f"Inconsistent Parquet schemas in '{config_name}' at {revision}")
        schema = current
    return schema


def _normalize_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([pa.field(f.name, _narrow(f.type)) for f in schema])


def _load_card(repo: str, token: Optional[str], revision: str) -> DatasetCard:
    """Read README at the same immutable revision as its Parquet shards."""
    from pathlib import Path

    path = hf_hub_download(repo, "README.md", repo_type="dataset", token=token, revision=revision)
    return DatasetCard(Path(path).read_text(encoding="utf-8"))


def _declared_schema(entry: dict) -> pa.Schema:
    from datasets import Features

    try:
        return _normalize_schema(Features._from_yaml_list(entry["features"]).arrow_schema)
    except Exception as exc:
        raise CardSchemaError(f"Invalid declared feature types: {exc}") from exc


def _features_yaml(schema: pa.Schema) -> List[dict]:
    """Serialise an Arrow schema the way ``datasets`` writes card YAML."""
    from datasets import Features

    features = Features.from_arrow_schema(schema)
    to_yaml = getattr(features, "_to_yaml_list", None)
    if to_yaml is None:  # pragma: no cover - guards a datasets API change
        raise CardSchemaError(
            "datasets.Features._to_yaml_list is gone, so this guard cannot "
            "serialise the corrected schema. Fix the card's dataset_info by hand "
            "and update iwac_common/card_sync.py."
        )
    return to_yaml()


def _declared(card: DatasetCard, config_name: str) -> Optional[dict]:
    infos = card.data.to_dict().get("dataset_info")
    if not infos:
        return None
    if isinstance(infos, dict):  # single-config cards are a bare mapping
        return infos if infos.get("config_name") in (None, config_name) else None
    return next((e for e in infos if e.get("config_name") == config_name), None)


def sync_card_features(
    repo: str,
    config_name: str,
    *,
    token: Optional[str] = None,
    console: Optional[Console] = None,
    expected_columns: Optional[Sequence[str]] = None,
    expected_schema: Optional[pa.Schema] = None,
    revision: Optional[str] = None,
    parquet_schema: Optional[pa.Schema] = None,
) -> None:
    """Verify, without writing, that one config's card features match its parquet.

    Raises :class:`CardSchemaError` on any mismatch. ``expected_columns`` and
    ``expected_schema`` additionally require the parquet on the Hub to be the
    frame the caller prepared, which catches a commit that landed something
    else. ``parquet_schema`` is the normalized footer schema when the caller has
    already read it at ``revision``; otherwise the footers are read here.
    """
    console = console or Console()
    token = resolve_hf_token(token)

    revision = revision or get_repo_revision(repo, token=token)
    schema = (
        _normalize_schema(parquet_schema) if parquet_schema is not None
        else _parquet_schema(repo, config_name, token, revision)
    )
    if expected_schema is not None and not schema.equals(_normalize_schema(expected_schema)):
        raise CardSchemaError(f"{repo}/{config_name}: published schema differs from the prepared dataset")
    actual = list(schema.names)
    if expected_columns is not None and actual != list(expected_columns):
        missing = [c for c in expected_columns if c not in actual]
        extra = [c for c in actual if c not in expected_columns]
        raise CardSchemaError(
            f"{repo} '{config_name}': the parquet on the Hub does not match the "
            f"frame that was pushed (missing {missing or 'none'}, unexpected "
            f"{extra or 'none'}). Do not re-push blindly; inspect the repo."
        )

    card = _load_card(repo, token, revision)
    entry = _declared(card, config_name)
    if entry is None or not entry.get("features"):
        console.print(
            f"[dim]ℹ Card for '{config_name}' declares no dataset_info features; "
            f"nothing can desync.[/dim]"
        )
        return

    declared = [f.get("name") for f in entry["features"]]
    if _declared_schema(entry).equals(schema):
        console.print(
            f"[green]✓[/green] Card schema matches the pushed parquet "
            f"({len(actual)} columns)."
        )
        return

    added = [c for c in actual if c not in declared]
    removed = [c for c in declared if c not in actual]
    detail = (
        f"{repo} '{config_name}': the card declares {len(declared)} columns but the "
        f"parquet has {len(actual)}"
        + (f"; undeclared: {', '.join(added)}" if added else "")
        + (f"; declared but absent: {', '.join(removed)}" if removed else "")
        + (
            "; column order or data types differ"
            if not added and not removed
            else ""
        )
    )
    console.print(
        f"[yellow]⚠[/yellow] {detail}.\n"
        f"[yellow]  load_dataset would raise CastError until the card is fixed.[/yellow]"
    )
    raise CardSchemaError(detail)


__all__ = ["CardSchemaError", "sync_card_features"]
