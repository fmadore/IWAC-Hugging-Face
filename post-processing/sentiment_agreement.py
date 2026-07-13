#!/usr/bin/env python3
"""
sentiment_agreement.py
======================

Inter-model agreement analysis for the three AI sentiment annotators
(Gemini, ChatGPT, Mistral) on the `articles` subset, plus optional
consensus columns pushed back to the Hub.

The three dimensions are ordinal 5-point scales:

- polarité      : Très négatif < Négatif < Neutre < Positif < Très positif
                  ("Non applicable" is treated as missing for scale metrics,
                  but still counts as a vote for the consensus label)
- centralité    : Non abordé < Marginal < Secondaire < Central < Très central
- subjectivité  : 1 (très objectif) … 5 (très subjectif)

Metrics reported per dimension:
- Pairwise Cohen's kappa (unweighted + quadratic-weighted)
- Krippendorff's alpha (nominal + interval), 3 raters, missing-tolerant
- Exact agreement rates (all three / pairwise)

Columns added with --push:
- consensus_polarite            majority label (>= 2 models), else ""
- consensus_centralite          majority label (>= 2 models), else ""
- consensus_subjectivite_score  median of available scores (float; with
                                exactly two raters the median is their mean,
                                so .5 values appear by design)
- sentiment_disagreement        pipe-joined dimensions in dispute
                                (polarite/centralite: no majority;
                                 subjectivite: score range >= 2)

Usage
-----
    python post-processing/sentiment_agreement.py [--source hub|csv] [--push]

The default run is report-only: metrics are printed and saved to
``analyses/output/sentiment_agreement_<config>.json``. Nothing is pushed
to the Hub unless --push is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import REPO_ROOT, ensure_hf_token, load_subset_dataframe, PRIVATE_REPO_ID  # noqa: E402

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

MODELS = ["gemini", "chatgpt", "mistral"]

POLARITY_ORDER = {
    "Très négatif": 1,
    "Négatif": 2,
    "Neutre": 3,
    "Positif": 4,
    "Très positif": 5,
}
CENTRALITY_ORDER = {
    "Non abordé": 1,
    "Marginal": 2,
    "Secondaire": 3,
    "Central": 4,
    "Très central": 5,
}

# Full ordinal scale for subjectivité (numeric 1-5); polarité/centralité get
# theirs from the *_ORDER mappings. Used to anchor weighted-kappa weights.
SUBJECTIVITY_SCALE = [1, 2, 3, 4, 5]

# (dimension key, column template, label→ordinal map or None for numeric)
DIMENSIONS = [
    ("polarite", "{m}_polarite", POLARITY_ORDER),
    ("centralite", "{m}_centralite_islam_musulmans", CENTRALITY_ORDER),
    ("subjectivite", "{m}_subjectivite_score", None),
]

OUTPUT_DIR = REPO_ROOT / "analyses" / "output"


# ---------------------------------------------------------------------------
# Agreement metrics (implemented locally — no scipy/sklearn dependency)
# ---------------------------------------------------------------------------


def cohen_kappa(
    a: np.ndarray,
    b: np.ndarray,
    weighted: bool = False,
    scale: Optional[List[int]] = None,
) -> Optional[float]:
    """Cohen's kappa on aligned integer ratings; quadratic weights if weighted.

    ``scale`` is the dimension's full ordered category set (e.g. 1..5).
    When given, the quadratic weight matrix is anchored to the theoretical
    scale min/max instead of the (max-min)² of the OBSERVED categories —
    the correct formulation when comparing kappa across pairs/dimensions
    where a pair may never use an extreme category. Note: because distances
    are computed from the true category VALUES and kappa is 1 - do/de, the
    normalizing constant cancels, so the numeric result matches the
    observed-anchored version (to float precision); passing the scale makes
    the anchoring explicit and robust to future weight-scheme changes.
    Unweighted kappa is unaffected: unused categories have zero marginals.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask].astype(int), b[mask].astype(int)
    if len(a) == 0:
        return None
    cats = sorted(scale) if scale is not None else sorted(set(a) | set(b))
    if len(cats) == 1:
        return 1.0  # both raters constant and identical
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    # Disagreement weights: identity for unweighted, quadratic for weighted.
    ci = np.array(cats, dtype=float)
    if weighted:
        w = (ci[:, None] - ci[None, :]) ** 2 / (ci.max() - ci.min()) ** 2
    else:
        w = 1.0 - np.eye(k)
    do, de = (w * obs).sum(), (w * exp).sum()
    if de == 0:
        return 1.0
    return float(1.0 - do / de)


def krippendorff_alpha(units: List[List[float]], metric: str = "interval") -> Optional[float]:
    """Krippendorff's alpha via the coincidence-matrix formulation.

    ``units`` is a list of per-item rating lists (missing already removed);
    items with fewer than 2 ratings are ignored, as per the method.
    """
    coincidence: Counter = Counter()
    values: set = set()
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[(vals[i], vals[j])] += 1.0 / (m - 1)
        values.update(vals)
    if not coincidence:
        return None
    vals_sorted = sorted(values)
    n_c = {c: sum(coincidence[(c, k)] for k in vals_sorted) for c in vals_sorted}
    n_total = sum(n_c.values())
    if n_total <= 1:
        return None

    def delta(c: float, k: float) -> float:
        if metric == "nominal":
            return 0.0 if c == k else 1.0
        return float((c - k) ** 2)  # interval

    d_obs = sum(coincidence[(c, k)] * delta(c, k) for c in vals_sorted for k in vals_sorted) / n_total
    d_exp = sum(
        n_c[c] * n_c[k] * delta(c, k) for c in vals_sorted for k in vals_sorted
    ) / (n_total * (n_total - 1))
    if d_exp == 0:
        return 1.0
    return float(1.0 - d_obs / d_exp)


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def to_ordinal(series: pd.Series, mapping: Optional[Dict[str, int]]) -> pd.Series:
    """Map raw column values onto the 1-5 ordinal scale (NaN when missing/unmapped)."""
    if mapping is None:
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    return s.map(mapping).astype(float)


def label_votes(row: pd.Series, cols: List[str]) -> List[str]:
    """Non-empty label votes for a row (keeps 'Non applicable' as a real vote)."""
    votes = []
    for c in cols:
        v = row[c]
        if pd.notna(v) and str(v).strip():
            votes.append(str(v).strip())
    return votes


def majority(votes: List[str]) -> str:
    """Label shared by >= 2 voters, else ''."""
    if not votes:
        return ""
    label, count = Counter(votes).most_common(1)[0]
    return label if count >= 2 else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inter-model sentiment agreement (Gemini/ChatGPT/Mistral) + consensus columns."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", default="articles", help="Subset with sentiment columns (articles)")
    parser.add_argument("--source", choices=["hub", "csv"], default="hub",
                        help="hub = live dataset (default); csv = local data/ mirror")
    parser.add_argument("--push", action="store_true",
                        help="Add consensus/disagreement columns and push to the Hub")
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]AI Sentiment Inter-Model Agreement[/bold cyan]\n"
        f"[dim]{' vs '.join(m.capitalize() for m in MODELS)} — "
        f"{args.repo} ({args.config})[/dim]",
        border_style="cyan",
    ))

    sentiment_cols = [tpl.format(m=m) for _, tpl, _ in DIMENSIONS for m in MODELS]
    needed = ["o:id"] + sentiment_cols

    token = ensure_hf_token(console=console) if (args.source == "hub" or args.push) else None
    df = load_subset_dataframe(
        args.repo, args.config, token=token, source=args.source,
        columns=needed if args.source == "csv" else None, console=console,
    )
    missing_cols = [c for c in sentiment_cols if c not in df.columns]
    if missing_cols:
        console.print(f"[red]✗[/red] Missing sentiment columns: {', '.join(missing_cols)}")
        return

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "repo": args.repo,
        "config": args.config,
        "source": args.source,
        "n_rows": int(len(df)),
        "models": MODELS,
        "dimensions": {},
    }

    consensus_frame = pd.DataFrame(index=df.index)
    disagreement_flags: Dict[str, pd.Series] = {}

    for dim_key, tpl, mapping in DIMENSIONS:
        cols = [tpl.format(m=m) for m in MODELS]
        ordinal = pd.DataFrame({m: to_ordinal(df[c], mapping) for m, c in zip(MODELS, cols)})
        n_full = int(ordinal.notna().all(axis=1).sum())
        n_any2 = int((ordinal.notna().sum(axis=1) >= 2).sum())

        # --- metrics ---
        # Anchor kappa weights to the dimension's full theoretical scale
        # (not just observed categories), so values are comparable across
        # pairs/dimensions.
        full_scale = sorted(mapping.values()) if mapping is not None else SUBJECTIVITY_SCALE
        pair_metrics = {}
        for m1, m2 in combinations(MODELS, 2):
            a, b = ordinal[m1].to_numpy(), ordinal[m2].to_numpy()
            both = ~(np.isnan(a) | np.isnan(b))
            exact = float(np.mean(a[both] == b[both])) if both.any() else None
            pair_metrics[f"{m1}-{m2}"] = {
                "n": int(both.sum()),
                "exact_agreement": exact,
                "kappa": cohen_kappa(a, b, weighted=False, scale=full_scale),
                "kappa_weighted_quadratic": cohen_kappa(a, b, weighted=True, scale=full_scale),
            }

        units = [row[~np.isnan(row)].tolist() for row in ordinal.to_numpy()]
        alpha_interval = krippendorff_alpha(units, metric="interval")
        alpha_nominal = krippendorff_alpha(units, metric="nominal")

        full_rows = ordinal.dropna()
        all3 = float((full_rows.nunique(axis=1) == 1).mean()) if len(full_rows) else None

        report["dimensions"][dim_key] = {
            "n_rated_by_all_3": n_full,
            "n_rated_by_2plus": n_any2,
            "exact_agreement_all_3": all3,
            "krippendorff_alpha_interval": alpha_interval,
            "krippendorff_alpha_nominal": alpha_nominal,
            "pairwise": pair_metrics,
        }

        # --- consensus columns ---
        if mapping is not None:
            raw_votes = df[cols].apply(lambda r: label_votes(r, cols), axis=1)
            consensus_frame[f"consensus_{dim_key}"] = raw_votes.apply(majority)
            disagreement_flags[dim_key] = raw_votes.apply(lambda v: len(v) >= 2) & (
                consensus_frame[f"consensus_{dim_key}"] == ""
            )
        else:
            # Median of the available scores. With exactly two raters the
            # median is the mean of the two values, so half-point scores
            # (e.g. 2.5, 3.5) appear by design — they are not an error.
            consensus_frame["consensus_subjectivite_score"] = ordinal.median(axis=1, skipna=True)
            spread = ordinal.max(axis=1) - ordinal.min(axis=1)
            disagreement_flags[dim_key] = (ordinal.notna().sum(axis=1) >= 2) & (spread >= 2)

        # --- display ---
        table = Table(title=f"Dimension: {dim_key}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_row("Rows rated by all 3", f"{n_full:,}")
        if all3 is not None:
            table.add_row("Exact agreement (all 3)", f"{all3:.1%}")
        if alpha_interval is not None:
            table.add_row("Krippendorff α (interval)", f"{alpha_interval:.3f}")
        if alpha_nominal is not None:
            table.add_row("Krippendorff α (nominal)", f"{alpha_nominal:.3f}")
        for pair, pm in pair_metrics.items():
            kw = pm["kappa_weighted_quadratic"]
            k = pm["kappa"]
            ex = pm["exact_agreement"]
            table.add_row(
                f"{pair}",
                f"κ={k:.3f}  κ_w={kw:.3f}  agree={ex:.1%}" if None not in (k, kw, ex) else "n/a",
            )
        console.print(table)

    # Combined disagreement column: pipe-joined dimensions in dispute
    disagreement = pd.Series([""] * len(df), index=df.index, dtype="object")
    for dim_key, _, _ in DIMENSIONS:
        flag = disagreement_flags[dim_key]
        disagreement = disagreement.where(~flag, disagreement + ("|" + dim_key))
    consensus_frame["sentiment_disagreement"] = disagreement.str.lstrip("|")

    n_disputed = int((consensus_frame["sentiment_disagreement"] != "").sum())
    report["consensus"] = {
        "columns": list(consensus_frame.columns),
        "rows_with_any_disagreement": n_disputed,
        "share_with_any_disagreement": n_disputed / len(df) if len(df) else None,
        "disagreement_breakdown": {
            dim: int(disagreement_flags[dim].sum()) for dim, _, _ in DIMENSIONS
        },
    }

    console.print(Panel(
        f"Rows with at least one disputed dimension: [bold]{n_disputed:,}[/bold] "
        f"({n_disputed / len(df):.1%})\n"
        + "\n".join(
            f"  [cyan]{dim}[/cyan]: {int(disagreement_flags[dim].sum()):,} disputed"
            for dim, _, _ in DIMENSIONS
        ),
        title="Consensus / Disagreement", border_style="blue",
    ))

    # --- save report ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / f"sentiment_agreement_{args.config}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    out_csv = OUTPUT_DIR / f"sentiment_consensus_{args.config}.csv"
    pd.concat([df[["o:id"]], consensus_frame], axis=1).to_csv(out_csv, index=False, encoding="utf-8")
    console.print(f"[green]✓[/green] Report: [cyan]{out_json}[/cyan]")
    console.print(f"[green]✓[/green] Consensus columns: [cyan]{out_csv}[/cyan]")

    # --- optional push ---
    if not args.push:
        console.print("[yellow]ℹ[/yellow] Report-only run. Use [bold]--push[/bold] to add "
                      "the consensus columns to the Hub dataset.")
        return

    if args.source != "hub":
        console.print("[red]✗[/red] --push requires --source hub (columns must align with live rows).")
        return

    from datasets import Dataset, load_dataset  # noqa: F401

    console.print("\n[bold cyan]Pushing consensus columns to the Hub...[/bold cyan]")
    ds = load_dataset(args.repo, name=args.config, split="train", token=token)
    updated = ds
    for col in consensus_frame.columns:
        if col in updated.column_names:
            updated = updated.remove_columns([col])
        values = consensus_frame[col]
        if values.dtype == object:
            updated = updated.add_column(col, [v if v else None for v in values.tolist()])
        else:
            updated = updated.add_column(col, [None if pd.isna(v) else float(v) for v in values.tolist()])
    updated.push_to_hub(
        repo_id=args.repo,
        config_name=args.config,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=(
            "Add AI sentiment consensus columns "
            f"({', '.join(consensus_frame.columns)}) from 3-model agreement analysis"
        ),
    )
    console.print("[green]✓[/green] Pushed.")


if __name__ == "__main__":
    main()
