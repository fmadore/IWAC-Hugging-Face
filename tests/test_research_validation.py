"""Behavioral checks for report-only DH validation tools on synthetic sources."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from iwac_pipeline.analyses.annotation_review import annotation_sample
from iwac_pipeline.analyses.corpus_coverage import coverage_report
from iwac_pipeline.analyses.reprint_candidates import evaluate_pairs, reprint_candidates
from iwac_pipeline.analyses.topic_review import parse_topk, topic_review
from iwac_common.sentiment_panel import latest_generation


@pytest.fixture
def corpus():
    rows = []
    for i in range(12):
        rows.append({"o:id": str(i + 1), "title": f"Source {i + 1}", "country": "Togo",
                     "newspaper": "Journal A", "newspaper_ids": "100", "language": "Français",
                     "pub_date": f"{2000 + i % 2}-01-01", "OCR": "private text must stay inside input",
                     "OCR_is_public": False, "lemma_nostop": "private lemma", "embedding_OCR": [1., 0.],
                     "lda_topic_id": 0, "lda_topic_prob": .8, "lda_topic_topk": "0:0.8|1:0.2",
                     "embedding_OCR_config_hash": "test-config", "nb_mots": 500,
                     "lda_model_name": "lda-sha256:" + "a" * 64})
    return pd.DataFrame(rows)


def test_coverage_reconciles_exclusions_and_missing_language(corpus):
    corpus.loc[0, "pub_date"] = "2000/2001"
    corpus.loc[1, "pub_date"] = "bad"
    corpus.loc[2, "language"] = None
    corpus.loc[3, "lemma_nostop"] = ""
    cells, ledger, summary = coverage_report(corpus, year_min=2000, year_max=2001,
        languages=["Français"], require=["lemma_nostop"])
    assert summary["documents"] == summary["included"] + summary["excluded"] == 12
    assert summary["excluded"] == 4
    assert ledger.loc[0, "exclusion_reasons"] == "missing_or_ambiguous_date"
    assert pd.isna(ledger.loc[0, "year"])
    assert ledger.loc[2, "language"] == "[missing]"
    assert cells[cells.dimension == "country"].n_documents.sum() == 12
    assert not any("private" in str(value) for value in ledger.to_numpy().flat)
    assert summary["stable_outlets"] == 1
    assert ledger.stable_outlet_cohort.sum() == 8


def test_missing_year_prevents_stable_outlet_and_ranges_not_assigned(corpus):
    corpus["pub_date"] = "2000"
    _, ledger, summary = coverage_report(corpus, year_min=2000, year_max=2001)
    assert summary["stable_outlets"] == 0
    assert not ledger.stable_outlet_cohort.any()


def test_parent_private_access_not_public_text(corpus):
    corpus["OCR_is_public"] = True
    corpus["o:is_public"] = False
    _, ledger, _ = coverage_report(corpus)
    assert set(ledger.access) == {"restricted_item"}


def test_stratified_sample_reproducible_blinded_and_weighted(corpus):
    models = latest_generation()
    for m in models:
        corpus[m.column("polarite")] = "Négatif"
    corpus[models[-1].column("polarite")] = "Positif"
    sheets, audit, summary = annotation_sample(corpus, per_stratum=3, challenge_size=4, seed=11)
    shuffled, shuffled_audit, _ = annotation_sample(corpus.sample(frac=1, random_state=7), per_stratum=3,
                                                    challenge_size=4, seed=11)
    for name in sheets:
        pd.testing.assert_frame_equal(sheets[name], shuffled[name])
    pd.testing.assert_frame_equal(audit, shuffled_audit)
    probability = audit[audit.sample_group == "stratified_probability"]
    challenge = audit[audit.sample_group == "challenge_nonprobability"]
    assert len(probability) == 3 and len(challenge) == 4
    assert set(probability["o:id"]).isdisjoint(challenge["o:id"])
    assert probability.design_weight.sum() == 12
    assert challenge.design_weight.isna().all()
    # Every sheet lists the same sources, in hash-rank order, interleaving both groups.
    assert audit["stable_rank"].is_monotonic_increasing
    for sheet in sheets.values():
        assert sheet["o:id"].tolist() == audit["o:id"].tolist()
    assert all(sheets["adjudication"].adjudicated_polarity.eq(""))
    assert summary["annotation_generation"] == models[0].generation


def test_annotator_sheets_are_blind_and_independent(corpus):
    models = latest_generation()
    corpus[models[0].column("polarite")] = "Négatif"
    corpus[models[-1].column("polarite")] = "Positif"
    sheets, audit, _ = annotation_sample(corpus, per_stratum=3, challenge_size=4, seed=11)
    assert set(sheets) == {"annotator_a", "annotator_b", "adjudication"}
    design = {"sample_group", "selection_reason", "stratum", "stratum_population", "stratum_sampled",
              "inclusion_probability", "design_weight", "stable_rank", "annotation_generation"}
    assert design.issubset(audit.columns)
    for name, sheet in sheets.items():
        assert design.isdisjoint(sheet.columns), name
        assert not any(col.endswith(("_polarite", "_subjectivite_score")) for col in sheet)
        assert "text_excerpt" not in sheet
        values = {str(v) for v in sheet.to_numpy().flat}
        assert "challenge_nonprobability" not in values and "stratified_probability" not in values
    # Each annotator sees only their own form; neither sees the other's or the adjudication fields.
    assert "polarity" in sheets["annotator_a"] and "annotator_b_polarity" not in sheets["annotator_a"]
    assert not any(col.startswith("adjudicated_") for col in sheets["annotator_b"])


def test_expert_challenge_unknown_ids_rejected(corpus):
    with pytest.raises(ValueError, match="absent from this snapshot"):
        annotation_sample(corpus, difficult_ids=["9999"])


def test_annotation_excerpt_requires_explicit_option(corpus):
    sheets, _, _ = annotation_sample(corpus, per_stratum=1, challenge_size=0, excerpt_chars=7)
    assert sheets["annotator_a"].text_excerpt.tolist() == ["private"]


def test_topic_representatives_and_borderline_are_disjoint_and_model_scoped(corpus):
    corpus.loc[0, ["lda_topic_prob", "lda_topic_topk"]] = [.99, "0:0.99|1:0.01"]
    corpus.loc[1, ["lda_topic_prob", "lda_topic_topk"]] = [.51, "0:0.51|1:0.49"]
    corpus.loc[2, "lda_model_name"] = "lda-sha256:" + "b" * 64
    corpus.loc[3, "lda_model_name"] = "lda_model"
    sheet, excluded, summary = topic_review(corpus, representatives=1, borderline=1)
    model_a = sheet[sheet.model_id == "lda-sha256:" + "a" * 64]
    assert model_a[model_a.review_role == "representative"]["o:id"].tolist() == ["1"]
    assert model_a[model_a.review_role == "borderline"]["o:id"].tolist() == ["2"]
    assert len(summary["model_ids"]) == 2
    assert excluded["o:id"].tolist() == ["4"]
    assert "text_excerpt" not in sheet
    assert all(sheet.approved_description.eq(""))


def test_topic_malformed_topk_never_becomes_borderline(corpus):
    corpus["lda_topic_topk"] = "0:0.8|0:0.2"
    sheet, _, _ = topic_review(corpus, representatives=1, borderline=5)
    assert len(sheet) == 1
    assert parse_topk("0:0.9|1:0.8") == []
    assert parse_topk("0:nan") == []
    assert parse_topk("0:0.7|1:0.2") == [(0, .7), (1, .2)]


def test_reprint_requires_lexical_support_and_never_marks_verified(corpus):
    corpus = corpus.head(4).copy()
    corpus.loc[0, "OCR"] = "Une association musulmane ouvre une école à Lomé"
    corpus.loc[1, "OCR"] = "Une association musulmane ouvre une école à Lomé"
    corpus.loc[2, "OCR"] = "Un long discours économique décrit une stratégie de développement régional"
    corpus.loc[3, "embedding_OCR"] = None
    pairs, exclusions, summary = reprint_candidates(corpus, min_jaccard=.7, block_size=2)
    assert pairs[["id_a", "id_b"]].to_records(index=False).tolist() == [("1", "2")]
    assert pairs.iloc[0].lexical_jaccard == 1
    assert pairs.iloc[0].status == "candidate_unverified"
    assert pairs.iloc[0].human_reprint == ""
    assert exclusions["o:id"].tolist() == ["4"]
    assert summary["pairs_checked_lexically"] == 3
    assert not any("excerpt" in c for c in pairs)


def test_reprint_date_filter_and_order_independence(corpus):
    corpus = corpus.head(4).copy()
    corpus["pub_date"] = ["2000-01-01", "2000-01-02", "2000-06-01", "2000"]
    pairs, exclusions, _ = reprint_candidates(corpus, max_days=7)
    assert len(pairs) == 1
    assert pairs.iloc[0].day_gap == 1
    assert exclusions["o:id"].tolist() == ["4"]
    shuffled, _, _ = reprint_candidates(corpus.sample(frac=1, random_state=4), max_days=7)
    pd.testing.assert_frame_equal(pairs, shuffled)


def test_reprint_rejects_mixed_or_invalid_vectors(corpus):
    corpus.at[0, "embedding_OCR"] = [1, 0, 0]
    with pytest.raises(ValueError, match="Mixed embedding dimensions"):
        reprint_candidates(corpus)
    corpus.at[0, "embedding_OCR"] = [np.nan, 0]
    _, excluded, _ = reprint_candidates(corpus)
    assert excluded.iloc[0]["o:id"] == "1"


def test_pair_evaluation_includes_missed_pairs_and_unknowns_are_not_negatives():
    predictions = pd.DataFrame([{"id_a": "1", "id_b": "2"}, {"id_a": "1", "id_b": "3"}])
    labels = pd.DataFrame([{"id_a": "2", "id_b": "1", "human_reprint": "yes"},
                           {"id_a": "1", "id_b": "3", "human_reprint": "no"},
                           {"id_a": "2", "id_b": "3", "human_reprint": "yes"},
                           {"id_a": "2", "id_b": "4", "human_reprint": ""}])
    result = evaluate_pairs(predictions, labels)
    assert result["precision_on_evaluated_pairs"] == .5
    assert result["recall_on_evaluated_pairs"] == .5
    assert result["adjudicated_pairs"] == 3
    assert result["unlabelled_pairs"] == 1
    with pytest.raises(ValueError, match="outside this input"):
        evaluate_pairs(predictions, labels, population_ids={"1", "2", "3"})


def test_offline_cli_records_unknown_revision_and_input_hash(tmp_path, corpus):
    source = tmp_path / "input.parquet"
    corpus.to_parquet(source, index=False)
    output = tmp_path / "reports"
    result = subprocess.run([sys.executable, "-m", "iwac_pipeline.analyses.corpus_coverage",
        "--input", str(source), "--year-min", "2000", "--year-max", "2001",
        "--output-dir", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "corpus_coverage.manifest.json").read_text())
    assert manifest["dataset"]["revision"] is None
    assert manifest["inputs"]["revision_status"] == "unknown"
    assert len(manifest["inputs"]["sha256"]) == 64
    ledger = pd.read_csv(output / "corpus_coverage_inclusion.csv")
    assert set(ledger.source_revision) == {"unknown"}
    assert "private text" not in (output / "corpus_coverage_inclusion.csv").read_text()


def test_reprint_missing_provenance_requires_explicit_exploratory_choice(corpus):
    corpus = corpus.drop(columns=["embedding_OCR_config_hash"])
    with pytest.raises(ValueError, match="provenance missing"):
        reprint_candidates(corpus)
    _, _, summary = reprint_candidates(corpus, allow_unverified_embeddings=True)
    assert summary["documents_with_unverified_embedding_provenance"] == len(corpus)


def test_reprint_mixed_known_configurations_always_fail(corpus):
    corpus.loc[0, "embedding_OCR_config_hash"] = "different-model"
    with pytest.raises(ValueError, match="Mixed embedding configurations"):
        reprint_candidates(corpus, allow_unverified_embeddings=True)


@pytest.mark.parametrize("module,arguments,report", [
    ("annotation_review", ["--per-stratum", "1", "--challenge-size", "0"], "annotation_review_annotator_a.csv"),
    ("topic_review", ["--representatives", "1", "--borderline", "1"], "topic_review_sheet.csv"),
    ("reprint_candidates", ["--top-k", "2"], "reprint_candidates_pairs.csv"),
])
def test_review_cli_metadata_only_end_to_end(tmp_path, corpus, module, arguments, report):
    source = tmp_path / "input.parquet"
    corpus.to_parquet(source, index=False)
    output = tmp_path / "reports"
    result = subprocess.run([sys.executable, "-m", f"iwac_pipeline.analyses.{module}",
        "--input", str(source), "--output-dir", str(output), *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    content = (output / report).read_text()
    assert "private text" not in content and "private lemma" not in content
    assert "source_revision" in content


def test_point_years_are_one_rule_for_every_analysis():
    from iwac_pipeline.analyses._research import date_exclusions, in_window, point_years
    years, precision = point_years(pd.Series(["2000-01-02", "2000-01", "2000", "2000/2001", "bad", None]))
    assert years.tolist()[:3] == [2000, 2000, 2000] and years.iloc[3:].isna().all()
    assert date_exclusions(precision)["by_precision"] == {"[missing]": 1, "other": 1, "range": 1}
    assert in_window(years, 2000, 2000).tolist() == [True, True, True, False, False, False]


def test_ids_file_filter_records_counts_and_rejects_foreign_cohorts(tmp_path, corpus):
    from types import SimpleNamespace
    from iwac_pipeline.analyses._research import apply_ids_filter, read_ids_file
    text = tmp_path / "ids.txt"
    text.write_text("# cohort\n1\n2\n\n3\n", encoding="utf-8")
    assert read_ids_file(text) == {"1", "2", "3"}
    kept = apply_ids_filter(corpus.copy(), SimpleNamespace(ids_file=text, ids_where=None))
    assert kept["o:id"].tolist() == ["1", "2", "3"]
    assert kept.attrs["ids_filter"]["rows_dropped"] == 9
    assert len(kept.attrs["ids_filter"]["sha256"]) == 64
    text.write_text("1\n999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from this input"):
        apply_ids_filter(corpus.copy(), SimpleNamespace(ids_file=text, ids_where=None))
    with pytest.raises(ValueError, match="needs a CSV"):
        read_ids_file(text, "stable_outlet_cohort")
    untouched = apply_ids_filter(corpus.copy(), SimpleNamespace(ids_file=None, ids_where=None))
    assert len(untouched) == 12 and untouched.attrs["ids_filter"] is None


def test_coverage_ledger_feeds_another_analysis_as_a_cohort(tmp_path, corpus):
    from types import SimpleNamespace
    from iwac_pipeline.analyses._research import apply_ids_filter
    corpus.loc[0, "pub_date"] = "2000/2001"
    _, ledger, summary = coverage_report(corpus, year_min=2000, year_max=2001)
    assert summary["date_exclusions"]["by_precision"] == {"range": 1}
    ledger.to_csv(tmp_path / "ledger.csv", index=False)
    cohort = apply_ids_filter(corpus.copy(), SimpleNamespace(ids_file=tmp_path / "ledger.csv",
                                                             ids_where="stable_outlet_cohort"))
    assert len(cohort) == int(ledger.stable_outlet_cohort.sum()) == 11


def test_reprint_shingles_only_candidate_documents(corpus):
    corpus = corpus.head(4).copy()
    corpus["embedding_OCR"] = [[1., 0.], [1., 0.], [0., 1.], [-1., 0.]]
    corpus.loc[0, "OCR"] = corpus.loc[1, "OCR"] = "Une association musulmane ouvre une école"
    pairs, _, summary = reprint_candidates(corpus, min_jaccard=.5)
    assert pairs[["id_a", "id_b"]].values.tolist() == [["1", "2"]]
    # Documents 3 and 4 have no cosine neighbour, so their text is never shingled.
    assert summary["documents_shingled"] == 2


BLOCK_PACKAGE = """
import runpy, sys
class Block:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name.split(".")[0] == "iwac_pipeline":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
if sys.argv[1] == "block":
    sys.meta_path.insert(0, Block)
script = sys.argv[2]
sys.argv = [script, "--help"]
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    assert exc.code in (0, None), exc.code
names = {m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
loaded = [n for n in names if "/analyses/" in n.replace("\\\\", "/")]
assert len(loaded) == len(set(loaded))
# One module object per analyses/ file: never both a plain and a package copy.
stems = [n.rsplit("/", 1)[-1] for n in loaded]
assert len(stems) == len(set(stems)), sorted(stems)
print("ok")
"""


@pytest.mark.parametrize("mode", ["block", "installed"])
@pytest.mark.parametrize("script", ["corpus_coverage", "annotation_review", "reprint_candidates",
                                    "similarity_calibration", "topic_review", "topic_prevalence",
                                    "keyness_bursts", "topic_sentiment", "entity_networks"])
def test_script_paths_run_without_the_installed_package(tmp_path, mode, script):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", BLOCK_PACKAGE, mode, str(root / "analyses" / f"{script}.py")],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
