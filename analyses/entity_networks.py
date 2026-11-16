#!/usr/bin/env python3
"""
entity_networks.py
==================

Subject co-occurrence networks from the IWAC authority file.

``articles.subject`` links authority records (persons, organisations,
places, events, subjects). This builds an undirected graph whose nodes are
those authority entities and whose edges weight how often two entities are
tagged on the same article — the actor/topic network of the corpus, ready for
Gephi.

Entities are resolved by id: ``articles.subject_ids`` carries the linked
authorities' ``o:id``, joined to ``index.o:id`` — two authorities sharing a
title stay two nodes. Rows without ids (a Hub revision older than the ids
columns) fall back to matching each ``subject`` value against ``index.Titre``
exactly only when the title is unambiguous.

Edge weights: ``co_articles`` counts articles tagging both entities; ``Weight``
(what Gephi reads) is Newman's fractional count, each article contributing
1/(m-1) to each of its pairs where m is its number of matched entities, so a
heavily catalogued article does not dominate through clique expansion. Edge
``pmi`` is computed over articles with at least one matched entity, so
"positive" means more than chance among catalogued articles. Degree reflects
cataloguing density as much as prominence. This describes catalogued
co-occurrence, not real-world social ties.

Outputs (analyses/output/, Gephi-ready):
- entity_nodes.csv   Id (index o:id), Label, Type, articles_count, first_year, last_year
- entity_edges.csv   Source, Target (index o:ids), Weight (fractional), co_articles, pmi, Type=Undirected
- entity_networks_summary.json, entity_networks.manifest.json

Never writes to the Hub.

Usage
-----
    python analyses/entity_networks.py [--source hub|local] [--revision SHA] [--min-node-count 10]
        [--min-edge-weight 3] [--country Bénin] [--year-from 2000] [--year-to 2020]
        [--types Personne Organisation] [--ids-file cohort.csv] [--top-edges 25]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, in_window, load_companion,
        load_input, load_subset_dataframe, point_years, save_reports, write_run_manifest,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import (  # noqa: E402
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, in_window, load_companion,
        load_input, load_subset_dataframe, point_years, save_reports, write_run_manifest,
    )

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_new_analyses.py)
# ---------------------------------------------------------------------------


def split_subjects(raw) -> Set[str]:
    """Deduplicated, stripped set of subjects from a pipe-separated field.
    Missing/NaN yields the empty set (a subject listed twice counts once)."""
    if raw is None or (not isinstance(raw, (list, tuple, set)) and pd.isna(raw)):
        return set()
    return {s.strip() for s in str(raw).split("|") if s.strip()}



def normalize_authority_id(value) -> str:
    """CSV numeric coercion must not turn the authority key 12 into '12.0'."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def authority_lookups(index: pd.DataFrame):
    """Return ID maps and an unambiguous-only title fallback map."""
    titles, kinds = {}, {}
    for row in index.to_dict("records"):
        oid = normalize_authority_id(row.get("o:id"))
        title = row.get("Titre")
        if not oid or pd.isna(title) or not str(title).strip():
            continue
        titles[oid] = str(title).strip()
        kind = row.get("Type")
        kinds[oid] = "" if pd.isna(kind) else str(kind).strip()
    counts = Counter(titles.values())
    by_title = {title: oid for oid, title in titles.items() if counts[title] == 1}
    return titles, kinds, by_title


def ambiguous_titles(id_to_title: Dict[str, str]) -> Set[str]:
    """Titles shared by several authority records (never resolved by label)."""
    return {title for title, count in Counter(id_to_title.values()).items() if count > 1}


def resolve_entities(
    row, id_to_title: Dict[str, str], title_to_id: Dict[str, str],
    ambiguous: Optional[Set[str]] = None,
) -> Tuple[Set[str], int]:
    """``(entity ids, subject tokens considered)`` for one article.

    ``subject_ids`` wins when the row has it; otherwise each ``subject`` label
    is matched only to a unique index title. Ambiguous homonyms remain unmatched.
    Loops pass ``ambiguous`` (from :func:`ambiguous_titles`) computed once.
    """
    ids = {normalize_authority_id(i) for i in split_subjects(row.get("subject_ids"))}
    if ids:
        return {i for i in ids if i in id_to_title}, len(ids)
    labels = split_subjects(row.get("subject"))
    if not labels:
        return set(), 0
    if ambiguous is None:
        ambiguous = ambiguous_titles(id_to_title)
    return {title_to_id[t] for t in labels if t in title_to_id and t not in ambiguous}, len(labels)


def cooccurrence(entity_sets: Iterable[Iterable[str]]) -> Tuple[Counter, Dict[Tuple[str, str], float], Counter]:
    """``(raw pair counts, Newman fractional weights, node article counts)``.

    Each article with m >= 2 entities adds 1 to the raw count of each of its
    m(m-1)/2 pairs but only 1/(m-1) to their fractional weight, so its total
    contribution to an entity's weighted degree is 1 however densely it is
    catalogued (Newman 2001).
    """
    raw: Counter = Counter()
    fractional: Dict[Tuple[str, str], float] = defaultdict(float)
    nodes: Counter = Counter()
    for entities in entity_sets:
        matched = sorted(set(entities))
        nodes.update(matched)
        if len(matched) < 2:
            continue
        share = 1.0 / (len(matched) - 1)
        for pair in combinations(matched, 2):
            raw[pair] += 1
            fractional[pair] += share
    return raw, dict(fractional), nodes


def pmi(pair_count: int, a_count: int, b_count: int, total: int) -> float:
    """Pointwise mutual information (log2) over articles for a co-occurring
    pair: log2( p(a,b) / (p(a) p(b)) ). Positive = the two entities appear
    together more than chance. ``total`` is the number of articles considered
    (the analysis passes articles with at least one matched entity).
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


NODE_COLUMNS = ["Id", "Label", "Type", "articles_count", "first_year", "last_year"]
EDGE_COLUMNS = ["Source", "Target", "Weight", "co_articles", "pmi", "Type"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subject co-occurrence networks from the IWAC authority file.")
    add_input_args(parser, source_default="hub", output_dir=OUTPUT_DIR, cohort=True, companion_index=True)
    parser.add_argument("--min-node-count", type=int, default=10,
                        help="Drop entities appearing in fewer than N articles")
    parser.add_argument("--min-edge-weight", type=int, default=3,
                        help="Drop co-occurrence edges below N shared articles (co_articles)")
    parser.add_argument("--country", default=None, help="Restrict to one country")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--types", nargs="+", default=None, metavar="TYPE",
                        help="Keep only authorities whose index Type is one of these (case-insensitive)")
    parser.add_argument("--top-edges", type=int, default=25, help="Rows in the console edge table")
    return parser


def filter_articles(df: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Country and point-year filters; undated rows stay only without a year filter."""
    years, precision = point_years(df["pub_date"] if "pub_date" in df else pd.Series(None, index=df.index))
    keep = pd.Series(True, index=df.index)
    if args.country:
        keep &= df["country"].astype(str).str.strip().eq(args.country)
    year_filter = args.year_from is not None or args.year_to is not None
    if year_filter:
        keep &= in_window(years, args.year_from, args.year_to)
    counts = {"rows_loaded": int(len(df)), "rows_after_filters": int(keep.sum()),
              "year_filter_active": year_filter, "dates": date_exclusions(precision)}
    if not year_filter:
        counts["dates"]["note"] = "No year filter: undated and range-dated rows stay in the network but carry no first/last year."
    return df.loc[keep], years.loc[keep], counts


def entity_type_filter(entity_type: Dict[str, str], types) -> Optional[Set[str]]:
    if not types:
        return None
    wanted = {t.strip().casefold() for t in types}
    return {oid for oid, kind in entity_type.items() if kind.casefold() in wanted}


def network_tables(entity_sets, years, id_to_title, entity_type, *, min_node_count, min_edge_weight):
    """Gephi node/edge frames; edges ranked by the fractional weight."""
    raw, fractional, node_articles = cooccurrence(entity_sets)
    tagged_articles = sum(1 for entities in entity_sets if entities)
    node_years: Dict[str, List[int]] = defaultdict(list)
    for entities, year in zip(entity_sets, years):
        if pd.notna(year):
            for entity in entities:
                node_years[entity].append(int(year))
    kept = {n for n, c in node_articles.items() if c >= min_node_count}
    nodes = pd.DataFrame([{
        "Id": n, "Label": id_to_title.get(n, n), "Type": entity_type.get(n, ""),
        "articles_count": node_articles[n],
        "first_year": min(node_years[n]) if node_years.get(n) else "",
        "last_year": max(node_years[n]) if node_years.get(n) else "",
    } for n in sorted(kept)], columns=NODE_COLUMNS)
    edges = pd.DataFrame([{
        "Source": a, "Target": b, "Weight": round(fractional[(a, b)], 6), "co_articles": count,
        # PMI conditional on articles with >= 1 matched entity, not the whole corpus.
        "pmi": round(pmi(count, node_articles[a], node_articles[b], tagged_articles), 3),
        "Type": "Undirected",
    } for (a, b), count in raw.items()
        if count >= min_edge_weight and a in kept and b in kept], columns=EDGE_COLUMNS)
    if not edges.empty:
        edges = edges.sort_values(["Weight", "co_articles", "Source", "Target"],
                                  ascending=[False, False, True, True]).reset_index(drop=True)
    return nodes, edges, tagged_articles


def weighted_degree(edges: pd.DataFrame) -> Counter:
    degree: Counter = Counter()
    for source, target, weight in zip(edges["Source"], edges["Target"], edges["Weight"]):
        degree[source] += weight
        degree[target] += weight
    return degree


def print_report(nodes, edges, args, id_to_title, entity_type) -> None:
    stats = Table(title="Network", box=box.ROUNDED)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green", justify="right")
    stats.add_row("Nodes", f"{len(nodes):,}")
    stats.add_row("Edges", f"{len(edges):,}")
    stats.add_row("Min node count", str(args.min_node_count))
    stats.add_row("Min shared articles", str(args.min_edge_weight))
    stats.add_row("Authority types", ", ".join(args.types) if args.types else "all")
    console.print(stats)
    degree = weighted_degree(edges)
    if degree:
        top = Table(title="Top entities by fractional weighted degree (reflects cataloguing density)",
                    box=box.ROUNDED)
        top.add_column("Entity", style="green", max_width=40)
        top.add_column("Type", style="cyan")
        top.add_column("Wt. degree", justify="right")
        for entity, value in degree.most_common(10):
            top.add_row(id_to_title.get(entity, entity), entity_type.get(entity, ""), f"{value:.1f}")
        console.print(top)
    if not edges.empty:
        et = Table(title=f"Strongest co-occurrences by fractional weight (top {args.top_edges})", box=box.ROUNDED)
        et.add_column("Source", style="green", max_width=32)
        et.add_column("Target", style="green", max_width=32)
        et.add_column("Weight", justify="right")
        et.add_column("Articles", justify="right")
        et.add_column("PMI", justify="right")
        for e in edges.head(args.top_edges).to_dict("records"):
            et.add_row(id_to_title.get(e["Source"], e["Source"]), id_to_title.get(e["Target"], e["Target"]),
                       f"{e['Weight']:.2f}", str(int(e["co_articles"])), f"{e['pmi']:.2f}")
        console.print(et)


def main() -> None:
    args = build_parser().parse_args()
    console.print(Panel.fit(
        "[bold cyan]Entity Co-occurrence Networks[/bold cyan]\n"
        "[dim]Subject authority join (articles.subject_ids ↔ index.o:id)[/dim]",
        border_style="cyan",
    ))
    articles = load_input(args, columns=["o:id", "subject", "subject_ids", "country", "pub_date"],
                          loader=load_subset_dataframe)
    articles = apply_ids_filter(articles, args)
    index = load_companion(args, articles, "index", columns=["o:id", "Titre", "Type"],
                           loader=load_subset_dataframe, path=args.index_input)
    id_to_title, entity_type, title_to_id = authority_lookups(index)
    ambiguous = ambiguous_titles(id_to_title)
    allowed = entity_type_filter(entity_type, args.types)
    console.print(f"[blue]→[/blue] Authority file: {len(id_to_title):,} entities"
                  + (f"; {len(allowed):,} of type {', '.join(args.types)}" if allowed is not None else ""))

    df, years, filter_counts = filter_articles(articles, args)
    entity_sets, considered_tokens, matched_tokens = [], 0, 0
    for row in df.to_dict("records"):
        entities, considered = resolve_entities(row, id_to_title, title_to_id, ambiguous)
        considered_tokens += considered
        matched_tokens += len(entities)
        entity_sets.append(entities if allowed is None else entities & allowed)
    nodes, edges, tagged = network_tables(entity_sets, years.tolist(), id_to_title, entity_type,
                                          min_node_count=args.min_node_count,
                                          min_edge_weight=args.min_edge_weight)
    match_rate = matched_tokens / considered_tokens if considered_tokens else 0.0
    console.print(f"[blue]→[/blue] {tagged:,} articles with ≥1 selected authority; "
                  f"subject→authority match rate {match_rate:.1%}")
    print_report(nodes, edges, args, id_to_title, entity_type)
    summary = {
        "filters": filter_counts, "types": args.types,
        "n_articles": int(len(df)), "n_with_matched_authorities": tagged,
        "n_subject_tokens": considered_tokens, "n_matched_subject_tokens": matched_tokens,
        "edge_weight": "Weight = Newman fractional count, sum over shared articles of 1/(m-1) with m the "
                       "article's matched (type-filtered) entities; co_articles = raw shared-article count",
        "edge_ranking": "Weight (fractional), then co_articles",
        "min_edge_weight_applies_to": "co_articles",
        "pmi_denominator": "articles with at least one matched (type-filtered) authority",
        "degree_caveat": "Weighted degree reflects cataloguing density (how many authorities are tagged per "
                         "article and how consistently) as much as an entity's prominence.",
        "interpretation": "Subject-tag co-occurrence is conditional on corpus selection and cataloguing; it does not establish social ties. Ambiguous title fallbacks are excluded.",
    }
    save_reports(args, articles, "entity_networks",
                 {"entity_nodes.csv": nodes, "entity_edges.csv": edges}, summary,
                 writer=write_run_manifest)
    console.print(f"\n[green]✓[/green] Gephi-ready CSVs in [cyan]{args.output_dir}[/cyan]")


if __name__ == "__main__":
    main()
