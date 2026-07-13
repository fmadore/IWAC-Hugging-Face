#!/usr/bin/env python3
"""
keyness_bursts.py
=================

Two classic corpus-DH analyses over the `articles` subset:

1. **Keyness (Dunning log-likelihood, G²)** — the most distinctive vocabulary
   of each country subcorpus and each decade, computed on ``lemma_nostop``
   with the same stopword sets as the LDA pipeline (geographic + generic
   noise removed, Islamic/domain terms kept, per the DH guidelines).

2. **Burst detection (Kleinberg 2-state automaton)** — periods when a
   ``subject`` term (controlled vocabulary, joins to the index subset)
   appears far above its corpus base rate: coverage spikes around events.

Outputs (analyses/output/):
- keyness_country.csv    country, rank, token, g2, count, rate_ratio
- keyness_decade.csv     decade, rank, token, g2, count, rate_ratio
- subject_bursts.csv     subject, start, end, weight, mentions_in_burst, total
- keyness_bursts_summary.json

Usage
-----
    python analyses/keyness_bursts.py [--source hub|csv] [--min-subject-total 30]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "post-processing"))

from _common import ensure_hf_token, load_subset_dataframe, PRIVATE_REPO_ID  # noqa: E402
from iwac_common.text_utils import simple_tokenize  # noqa: E402
from lda_topic_modeling.constants import (  # noqa: E402
    DOMAIN_STOPWORDS,
    LDA_GEO_STOPWORDS,
    LDA_GENERIC_STOPWORDS,
)

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
OUTPUT_DIR = REPO_ROOT / "analyses" / "output"


# ---------------------------------------------------------------------------
# Keyness — Dunning log-likelihood
# ---------------------------------------------------------------------------


def dunning_g2(a: int, b: int, total_a: int, total_b: int) -> float:
    """Signed G² for token overrepresentation in corpus A vs corpus B."""
    if a == 0 or total_a == 0:
        return 0.0
    e1 = total_a * (a + b) / (total_a + total_b)
    e2 = total_b * (a + b) / (total_a + total_b)
    g2 = 0.0
    if a > 0 and e1 > 0:
        g2 += a * math.log(a / e1)
    if b > 0 and e2 > 0:
        g2 += b * math.log(b / e2)
    g2 *= 2.0
    # Sign: positive when overrepresented in A
    return g2 if (a / total_a) > ((b / total_b) if total_b else 0) else -g2


def keyness_for_slices(
    slice_tokens: dict[str, Counter],
    top_n: int,
    min_count: int,
) -> pd.DataFrame:
    """Top-N overrepresented tokens per slice vs. all other slices pooled."""
    totals = {s: sum(c.values()) for s, c in slice_tokens.items()}
    grand: Counter = Counter()
    for c in slice_tokens.values():
        grand.update(c)
    grand_total = sum(totals.values())

    rows = []
    for s, counter in slice_tokens.items():
        total_a = totals[s]
        total_b = grand_total - total_a
        scored = []
        for tok, a in counter.items():
            if a < min_count:
                continue
            b = grand[tok] - a
            g2 = dunning_g2(a, b, total_a, total_b)
            if g2 > 0:
                rate_a = a / total_a
                rate_b = (b / total_b) if total_b else 0.0
                ratio = rate_a / rate_b if rate_b else float("inf")
                scored.append((tok, g2, a, ratio))
        scored.sort(key=lambda x: x[1], reverse=True)
        for rank, (tok, g2, a, ratio) in enumerate(scored[:top_n], 1):
            rows.append({"slice": s, "rank": rank, "token": tok,
                         "g2": round(g2, 2), "count": a,
                         "rate_ratio": round(ratio, 2) if math.isfinite(ratio) else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bursts — Kleinberg's 2-state automaton (batch, document-stream variant)
# ---------------------------------------------------------------------------


def kleinberg_bursts(
    r: np.ndarray, d: np.ndarray, years: np.ndarray, s: float = 2.0, gamma: float = 1.0
) -> list[dict]:
    """Detect burst intervals for one term.

    r[t] = docs mentioning the term in year t; d[t] = all docs in year t.
    Two states: base rate p0 = R/D and burst rate p1 = s*p0. Entering the
    burst state costs gamma*ln(T); staying or leaving is free. Returns the
    maximal state-1 intervals with their total weight (cost saved vs. state 0).
    """
    T = len(r)
    R, D = float(r.sum()), float(d.sum())
    if T < 2 or R == 0 or D == 0:
        return []
    p0 = R / D
    p1 = min(s * p0, 0.9999)
    if p1 <= p0:
        return []

    def sigma(p: float, rt: float, dt: float) -> float:
        if dt == 0:
            return 0.0
        return -(rt * math.log(p) + (dt - rt) * math.log(1.0 - p))

    trans = gamma * math.log(T)
    INF = float("inf")
    cost = np.full((T, 2), INF)
    back = np.zeros((T, 2), dtype=int)
    cost[0, 0] = sigma(p0, r[0], d[0])
    cost[0, 1] = trans + sigma(p1, r[0], d[0])
    for t in range(1, T):
        for q in (0, 1):
            emit = sigma(p1 if q else p0, r[t], d[t])
            stay = cost[t - 1, q]
            move = cost[t - 1, 1 - q] + (trans if q == 1 else 0.0)
            if stay <= move:
                cost[t, q] = stay + emit
                back[t, q] = q
            else:
                cost[t, q] = move + emit
                back[t, q] = 1 - q

    # Backtrack optimal path
    q = int(np.argmin(cost[T - 1]))
    path = [q]
    for t in range(T - 1, 0, -1):
        q = int(back[t, q])
        path.append(q)
    path.reverse()

    bursts = []
    t = 0
    while t < T:
        if path[t] == 1:
            start = t
            weight = 0.0
            mentions = 0
            while t < T and path[t] == 1:
                weight += sigma(p0, r[t], d[t]) - sigma(p1, r[t], d[t])
                mentions += int(r[t])
                t += 1
            bursts.append({
                "start": int(years[start]), "end": int(years[t - 1]),
                "weight": round(weight, 2), "mentions_in_burst": mentions,
            })
        else:
            t += 1
    return bursts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyness (Dunning G²) + subject burst detection.")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", default="articles")
    parser.add_argument("--source", choices=["hub", "csv"], default="hub")
    parser.add_argument("--top-n", type=int, default=25, help="Keywords kept per slice")
    parser.add_argument("--min-count", type=int, default=10, help="Min token count in a slice")
    parser.add_argument("--min-subject-total", type=int, default=30,
                        help="Min total mentions for a subject to enter burst detection")
    parser.add_argument("--burst-s", type=float, default=2.0, help="Kleinberg burst rate multiplier")
    parser.add_argument("--burst-gamma", type=float, default=1.0, help="Kleinberg entry cost factor")
    parser.add_argument("--year-min", type=int, default=1900)
    parser.add_argument("--year-max", type=int, default=2030)
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Keyness & Burst Detection[/bold cyan]\n"
        "[dim]Dunning G² per country/decade · Kleinberg bursts on subjects[/dim]",
        border_style="cyan",
    ))

    token = ensure_hf_token(console=console) if args.source == "hub" else None
    df = load_subset_dataframe(
        args.repo, args.config, token=token, source=args.source,
        columns=["o:id", "lemma_nostop", "pub_date", "country", "subject", "language"],
        console=console,
    )

    years = pd.to_numeric(df["pub_date"].astype(str).str[:4], errors="coerce")
    df = df.assign(year=years)
    valid_year = df["year"].between(args.year_min, args.year_max)

    stopwords = set(DOMAIN_STOPWORDS) | LDA_GEO_STOPWORDS | LDA_GENERIC_STOPWORDS

    # ---------------- Keyness ----------------
    is_french = df["language"].isna() | (df["language"] == "Français")
    text_rows = df[is_french & df["lemma_nostop"].notna()]
    console.print(f"[blue]→[/blue] Keyness corpus: {len(text_rows):,} French articles")

    country_tokens: dict[str, Counter] = defaultdict(Counter)
    decade_tokens: dict[str, Counter] = defaultdict(Counter)
    for _, row in text_rows.iterrows():
        # simple_tokenize lowercases, so tokenization now matches the LDA preprocessing
        # (previously keyness skipped lowercasing and case-split tokens LDA merges).
        toks = simple_tokenize(row["lemma_nostop"], stopwords)
        if not toks:
            continue
        c = row["country"]
        if pd.notna(c) and str(c).strip():
            country_tokens[str(c).strip()].update(toks)
        y = row["year"]
        if pd.notna(y) and args.year_min <= y <= args.year_max:
            decade_tokens[f"{int(y // 10 * 10)}s"].update(toks)

    keyness_country = keyness_for_slices(country_tokens, args.top_n, args.min_count)
    keyness_decade = keyness_for_slices(decade_tokens, args.top_n, args.min_count)

    # ---------------- Bursts ----------------
    subj_rows = df[valid_year & df["subject"].notna()]
    per_year_docs = subj_rows.groupby(subj_rows["year"].astype(int)).size()
    year_index = np.array(sorted(per_year_docs.index))
    d_arr = per_year_docs.reindex(year_index).to_numpy(dtype=float)

    subject_year: dict[str, Counter] = defaultdict(Counter)
    for _, row in subj_rows.iterrows():
        y = int(row["year"])
        for s_term in str(row["subject"]).split("|"):
            s_term = s_term.strip()
            if s_term:
                subject_year[s_term][y] += 1

    console.print(
        f"[blue]→[/blue] Burst detection: {len(subject_year):,} subjects, "
        f"{year_index.min()}–{year_index.max()} "
        f"(threshold: ≥{args.min_subject_total} total mentions)"
    )

    burst_rows = []
    for subject, counts in subject_year.items():
        total = sum(counts.values())
        if total < args.min_subject_total:
            continue
        r_arr = np.array([counts.get(int(y), 0) for y in year_index], dtype=float)
        for b in kleinberg_bursts(r_arr, d_arr, year_index, s=args.burst_s, gamma=args.burst_gamma):
            burst_rows.append({"subject": subject, **b, "total_mentions": total})
    bursts_df = pd.DataFrame(burst_rows).sort_values("weight", ascending=False) if burst_rows else pd.DataFrame()

    # ---------------- Outputs ----------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keyness_country.to_csv(OUTPUT_DIR / "keyness_country.csv", index=False, encoding="utf-8")
    keyness_decade.to_csv(OUTPUT_DIR / "keyness_decade.csv", index=False, encoding="utf-8")
    bursts_df.to_csv(OUTPUT_DIR / "subject_bursts.csv", index=False, encoding="utf-8")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "config": args.config,
        "source": args.source,
        "keyness": {
            "countries": sorted(country_tokens),
            "decades": sorted(decade_tokens),
            "stopword_count": len(stopwords),
            "top_n": args.top_n,
        },
        "bursts": {
            "subjects_tested": sum(1 for c in subject_year.values() if sum(c.values()) >= args.min_subject_total),
            "bursts_found": int(len(bursts_df)),
            "s": args.burst_s, "gamma": args.burst_gamma,
        },
    }
    with open(OUTPUT_DIR / "keyness_bursts_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- Report ----------------
    kc = Table(title="Distinctive vocabulary per country (top 8 by G²)", box=box.ROUNDED)
    kc.add_column("Country", style="cyan")
    kc.add_column("Keywords", style="green")
    for country in sorted(country_tokens):
        top = keyness_country[keyness_country["slice"] == country].head(8)["token"]
        kc.add_row(country, ", ".join(top))
    console.print(kc)

    kd = Table(title="Distinctive vocabulary per decade (top 8 by G²)", box=box.ROUNDED)
    kd.add_column("Decade", style="cyan")
    kd.add_column("Keywords", style="green")
    for decade in sorted(decade_tokens):
        top = keyness_decade[keyness_decade["slice"] == decade].head(8)["token"]
        kd.add_row(decade, ", ".join(top))
    console.print(kd)

    if not bursts_df.empty:
        bt = Table(title="Strongest subject bursts (Kleinberg)", box=box.ROUNDED)
        bt.add_column("Subject", style="green", max_width=46)
        bt.add_column("Period", style="cyan", justify="center")
        bt.add_column("Weight", justify="right")
        bt.add_column("Mentions", justify="right")
        for _, r in bursts_df.head(20).iterrows():
            period = f"{r.start}" if r.start == r.end else f"{r.start}–{r.end}"
            bt.add_row(r.subject, period, f"{r.weight:,.0f}", f"{r.mentions_in_burst:,}")
        console.print(bt)

    console.print(f"\n[green]✓[/green] Outputs in [cyan]{OUTPUT_DIR}[/cyan]")


if __name__ == "__main__":
    main()
