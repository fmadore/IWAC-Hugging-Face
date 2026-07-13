#!/usr/bin/env python3
"""
entity_networks.py
==================

Subject co-occurrence networks from the IWAC authority file.

``articles.subject`` is a controlled vocabulary: each pipe-separated value
matches an ``index.Titre`` exactly (persons, organisations, places, events,
subjects). This builds an undirected graph whose nodes are those authority
entities and whose edges weight how often two entities are tagged on the same
article — the actor/topic network of the corpus, ready for Gephi.

For each article: split ``subject`` on ``|``, deduplicate, keep values that
match an ``index.Titre``, and attach the entity ``Type`` from the index plus
the article's country/year. Edge weight = number of co-occurring articles;
edge ``pmi`` = pointwise mutual information over articles (association
strength independent of raw frequency).

Outputs (analyses/output/, Gephi-ready):
- entity_nodes.csv   Id, Label, Type, articles_count, first_year, last_year
- entity_edges.csv   Source, Target, Weight, pmi, Type=Undirected

Never writes to the Hub.

Usage
-----
    python analyses/entity_networks.py [--source hub|csv] [--min-node-count 10]
        [--min-edge-weight 3] [--country Bénin] [--year-from 2000] [--year-to 2020]
        [--top-edges 25]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "post-processing"))

from _common import ensure_hf_token, load_subset_dataframe, PRIVATE_REPO_ID  # noqa: E402

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()
OUTPUT_DIR = REPO_ROOT / "analyses" / "output"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_new_analyses.py)
# ---------------------------------------------------------------------------


def split_subjects(raw) -> Set[str]:
    """Deduplicated, stripped set of subjects from a pipe-separated field.
    Missing/NaN yields the empty set (a subject listed twice counts once)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    return {s.strip() for s in str(raw).split("|") if s.strip()}


def parse_year(pub_date) -> Optional[int]:
    """Year from the YYYY prefix of a pub_date; None if unparsable."""
    if pub_date is None or (isinstance(pub_date, float) and pd.isna(pub_date)):
        return None
    s = str(pub_date).strip()[:4]
    return int(s) if s.isdigit() and len(s) == 4 else None


def pmi(pair_count: int, a_count: int, b_count: int, total: int) -> float:
    """Pointwise mutual information (log2) over articles for a co-occurring
    pair: log2( p(a,b) / (p(a) p(b)) ). Positive = the two entities appear
    together more than chance. ``total`` is the number of articles considered.
    """
    if pair_count <= 0 or a_count <= 0 or b_count <= 0 or total <= 0:
        return float("nan")
    p_ab = pair_count / total
    p_a = a_count / total
    p_b = b_count / total
    return math.log2(p_ab / (p_a * p_b))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Subject co-occurrence networks from the IWAC authority file.")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--source", choices=["hub", "csv"], default="hub")
    parser.add_argument("--min-node-count", type=int, default=10,
                        help="Drop entities appearing in fewer than N articles")
    parser.add_argument("--min-edge-weight", type=int, default=3,
                        help="Drop co-occurrence edges below N shared articles")
    parser.add_argument("--country", default=None, help="Restrict to one country")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--top-edges", type=int, default=25, help="Rows in the console edge table")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Entity Co-occurrence Networks[/bold cyan]\n"
        "[dim]Subject authority join (articles.subject ↔ index.Titre)[/dim]",
        border_style="cyan",
    ))

    token = ensure_hf_token(console=console) if args.source == "hub" else None
    articles = load_subset_dataframe(
        args.repo, "articles", token=token, source=args.source,
        columns=["o:id", "subject", "country", "pub_date"], console=console,
    )
    index = load_subset_dataframe(
        args.repo, "index", token=token, source=args.source,
        columns=["Titre", "Type"], console=console,
    )

    # Authority lookup: Titre -> Type.
    entity_type: Dict[str, str] = {}
    for _, r in index.iterrows():
        titre = str(r.get("Titre", "")).strip()
        if titre and titre not in entity_type:
            entity_type[titre] = str(r.get("Type", "") or "").strip()
    console.print(f"[blue]→[/blue] Authority file: {len(entity_type):,} entities")

    # Optional filters.
    df = articles
    if args.country:
        df = df[df["country"].astype(str).str.strip() == args.country]
    years = df["pub_date"].map(parse_year)
    if args.year_from is not None:
        df = df[years.fillna(-1).astype(int) >= args.year_from]
        years = df["pub_date"].map(parse_year)
    if args.year_to is not None:
        df = df[years.fillna(10**9).astype(int) <= args.year_to]

    # Per-article matched entities.
    node_articles: Counter = Counter()
    node_years: Dict[str, List[int]] = defaultdict(list)
    pair_counts: Counter = Counter()
    matched_articles = 0
    total_subject_tokens = 0
    matched_subject_tokens = 0

    for _, row in df.iterrows():
        subs = split_subjects(row.get("subject"))
        total_subject_tokens += len(subs)
        matched = sorted(s for s in subs if s in entity_type)
        matched_subject_tokens += len(matched)
        if not matched:
            continue
        matched_articles += 1
        yr = parse_year(row.get("pub_date"))
        for s in matched:
            node_articles[s] += 1
            if yr is not None:
                node_years[s].append(yr)
        for a, b in combinations(matched, 2):
            pair_counts[(a, b)] += 1

    match_rate = (matched_subject_tokens / total_subject_tokens) if total_subject_tokens else 0.0
    console.print(
        f"[blue]→[/blue] {matched_articles:,} articles with ≥1 authority subject; "
        f"subject→authority match rate {match_rate:.1%}"
    )

    # Nodes passing the frequency threshold.
    kept_nodes = {n for n, c in node_articles.items() if c >= args.min_node_count}
    total_articles = max(1, matched_articles)

    node_rows = []
    for n in sorted(kept_nodes):
        yrs = node_years.get(n, [])
        node_rows.append({
            "Id": n, "Label": n, "Type": entity_type.get(n, ""),
            "articles_count": node_articles[n],
            "first_year": min(yrs) if yrs else "",
            "last_year": max(yrs) if yrs else "",
        })
    nodes_df = pd.DataFrame(node_rows)

    # Edges among kept nodes passing the weight threshold.
    edge_rows = []
    for (a, b), w in pair_counts.items():
        if w < args.min_edge_weight or a not in kept_nodes or b not in kept_nodes:
            continue
        edge_rows.append({
            "Source": a, "Target": b, "Weight": w,
            "pmi": round(pmi(w, node_articles[a], node_articles[b], total_articles), 3),
            "Type": "Undirected",
        })
    edges_df = pd.DataFrame(edge_rows).sort_values("Weight", ascending=False) if edge_rows else pd.DataFrame(
        columns=["Source", "Target", "Weight", "pmi", "Type"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(OUTPUT_DIR / "entity_nodes.csv", index=False, encoding="utf-8")
    edges_df.to_csv(OUTPUT_DIR / "entity_edges.csv", index=False, encoding="utf-8")

    # Weighted degree (no networkx dependency).
    wdeg: Counter = Counter()
    for _, e in edges_df.iterrows():
        wdeg[e["Source"]] += e["Weight"]
        wdeg[e["Target"]] += e["Weight"]

    stats = Table(title="Network", box=box.ROUNDED)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green", justify="right")
    stats.add_row("Nodes", f"{len(nodes_df):,}")
    stats.add_row("Edges", f"{len(edges_df):,}")
    stats.add_row("Min node count", str(args.min_node_count))
    stats.add_row("Min edge weight", str(args.min_edge_weight))
    console.print(stats)

    if wdeg:
        top = Table(title="Top entities by weighted degree", box=box.ROUNDED)
        top.add_column("Entity", style="green", max_width=40)
        top.add_column("Type", style="cyan")
        top.add_column("Wt. degree", justify="right")
        for entity, deg in wdeg.most_common(10):
            top.add_row(entity, entity_type.get(entity, ""), str(int(deg)))
        console.print(top)

    if not edges_df.empty:
        et = Table(title=f"Strongest co-occurrences (top {args.top_edges})", box=box.ROUNDED)
        et.add_column("Source", style="green", max_width=32)
        et.add_column("Target", style="green", max_width=32)
        et.add_column("Weight", justify="right")
        et.add_column("PMI", justify="right")
        for _, e in edges_df.head(args.top_edges).iterrows():
            et.add_row(e["Source"], e["Target"], str(int(e["Weight"])), f"{e['pmi']:.2f}")
        console.print(et)

    console.print(f"\n[green]✓[/green] Gephi-ready CSVs in [cyan]{OUTPUT_DIR}[/cyan]")


if __name__ == "__main__":
    main()
