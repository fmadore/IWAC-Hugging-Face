"""Post-push guard: keep the Hub dataset card's declared schema in step with the
parquet that was actually pushed.

**Why this exists.** ``push_to_hub`` rewrites the card's ``dataset_info`` byte
sizes for the config it just pushed but, observed twice on 2026-08-06 with
``datasets`` 5.0.1, does *not* rewrite that config's ``features`` list. A push
that adds or removes a column therefore leaves the card declaring the old schema,
and ``load_dataset`` then raises::

    CastError: Couldn't cast … because column names don't match

That is a hard read failure, not a cosmetic drift: the subset becomes unloadable
for every consumer — and for this pipeline's own next run, since ``hub_merge``
loads the Hub copy before merging. It happened to ``articles`` on the private
mirror and then, from ``publish_public.py``, to the public *citable* dataset.

**Why it repairs rather than just aborting.** Every other rail in this repo
aborts *before* touching the Hub, so aborting protects the data. Here the push has
already happened; refusing to continue would leave the dataset broken. So this
checks, rewrites the one stale ``features`` list from the parquet that is now on
the Hub, and re-verifies. It raises only when the mismatch survives that.

Deliberately narrow: it replaces one config's ``features`` and nothing else —
never the byte sizes (``push_to_hub`` computes those correctly) and never another
config's entry, which is asserted before the card is written.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import DatasetCard, HfApi, HfFileSystem
from rich.console import Console

from .hub import resolve_hf_token


class CardSchemaError(RuntimeError):
    """The card's declared schema disagrees with the pushed parquet and could not
    be repaired. The subset is unloadable until the card is fixed."""


def _narrow(dtype: pa.DataType) -> pa.DataType:
    """``large_string`` → ``string``, recursively through lists.

    The two are the same logical type, ``string`` is what every hand-written
    entry in these cards uses, and it matches the ``num_bytes`` ``push_to_hub``
    records (which assumes 4-byte offsets). Normalising *both* sides before
    comparing is also what stops this guard from "repairing" on every push.
    """
    if pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.list_(_narrow(dtype.value_type))
    return dtype


def _parquet_schema(repo: str, config_name: str, token: Optional[str]) -> pa.Schema:
    """Arrow schema of the config's parquet on the Hub, read from the footer only
    (no shard download). All shards of a config share one schema."""
    fs = HfFileSystem(token=token)
    shards = sorted(fs.glob(f"datasets/{repo}/{config_name}/*.parquet"))
    if not shards:
        raise CardSchemaError(
            f"No parquet found for '{config_name}' in {repo}; cannot verify the card."
        )
    with fs.open(shards[0], "rb") as handle:
        schema = pq.ParquetFile(handle).schema_arrow
    return pa.schema([pa.field(f.name, _narrow(f.type)) for f in schema])


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


def _entry_index(card: DatasetCard, config_name: str) -> Optional[int]:
    infos = card.data.get("dataset_info")
    if not isinstance(infos, list):
        return None
    return next(
        (i for i, e in enumerate(infos) if e.get("config_name") == config_name), None
    )


def sync_card_features(
    repo: str,
    config_name: str,
    *,
    token: Optional[str] = None,
    console: Optional[Console] = None,
    repair: bool = True,
    expected_columns: Optional[Sequence[str]] = None,
) -> bool:
    """Verify — and by default repair — the card's declared features for one config.

    Returns ``True`` when the card already matched, ``False`` when it was
    repaired. Raises :class:`CardSchemaError` if the mismatch survives the repair
    or ``repair=False``.

    ``expected_columns`` is optional belt-and-braces: when given, the parquet on
    the Hub must also carry exactly those columns, which catches a push that
    landed a different frame than the caller thinks it did.
    """
    console = console or Console()
    token = resolve_hf_token(token)

    schema = _parquet_schema(repo, config_name, token)
    actual = list(schema.names)
    if expected_columns is not None and actual != list(expected_columns):
        missing = [c for c in expected_columns if c not in actual]
        extra = [c for c in actual if c not in expected_columns]
        raise CardSchemaError(
            f"{repo} '{config_name}': the parquet on the Hub does not match the "
            f"frame that was pushed (missing {missing or 'none'}, unexpected "
            f"{extra or 'none'}). Do not re-push blindly; inspect the repo."
        )

    card = DatasetCard.load(repo, repo_type="dataset", token=token)
    entry = _declared(card, config_name)
    if entry is None or not entry.get("features"):
        console.print(
            f"[dim]ℹ Card for '{config_name}' declares no dataset_info features; "
            f"nothing can desync.[/dim]"
        )
        return True

    declared = [f.get("name") for f in entry["features"]]
    if declared == actual:
        console.print(
            f"[green]✓[/green] Card schema matches the pushed parquet "
            f"({len(actual)} columns)."
        )
        return True

    added = [c for c in actual if c not in declared]
    removed = [c for c in declared if c not in actual]
    detail = (
        f"{repo} '{config_name}': the card declares {len(declared)} columns but the "
        f"parquet has {len(actual)}"
        + (f"; undeclared: {', '.join(added)}" if added else "")
        + (f"; declared but absent: {', '.join(removed)}" if removed else "")
        + (
            "; same columns in a different order"
            if not added and not removed
            else ""
        )
    )
    console.print(
        f"[yellow]⚠[/yellow] {detail}.\n"
        f"[yellow]  load_dataset would raise CastError until the card is fixed.[/yellow]"
    )
    if not repair:
        raise CardSchemaError(detail)

    index = _entry_index(card, config_name)
    if index is None:
        raise CardSchemaError(
            f"{detail}. The card's dataset_info is not a list, so this guard "
            f"cannot rewrite one entry; fix it by hand."
        )
    infos = card.data["dataset_info"]
    others_before = [e for i, e in enumerate(infos) if i != index]
    infos[index]["features"] = _features_yaml(schema)
    if [e for i, e in enumerate(infos) if i != index] != others_before:
        raise CardSchemaError(  # pragma: no cover - defensive
            f"{detail}. Rewriting '{config_name}' altered another config's entry; "
            f"refusing to write the card."
        )

    HfApi(token=token).upload_file(
        path_or_fileobj=card.content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=(
            f"Card: declare the {config_name} schema pushed in the preceding commit "
            f"({len(actual)} columns) — fixes load_dataset CastError"
        ),
    )

    # Re-verify against the Hub rather than trusting the local object.
    reloaded = _declared(
        DatasetCard.load(repo, repo_type="dataset", token=token), config_name
    )
    names = [f.get("name") for f in (reloaded or {}).get("features", [])]
    if names != actual:
        raise CardSchemaError(
            f"{detail}. The card was rewritten but still does not match "
            f"({len(names)} columns declared). Fix it by hand before anyone loads "
            f"this subset."
        )
    console.print(
        f"[green]✓[/green] Card repaired: '{config_name}' now declares "
        f"{len(actual)} columns and loads cleanly."
    )
    return False


__all__ = ["CardSchemaError", "sync_card_features"]
