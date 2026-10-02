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
                     "embedding_OCR_config_hash": "test-config",
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
    sheet, audit, summary = annotation_sample(corpus, per_stratum=3, challenge_size=4, seed=11)
    shuffled, _, _ = annotation_sample(corpus.sample(frac=1, random_state=7), per_stratum=3,
                                        challenge_size=4, seed=11)
    pd.testing.assert_frame_equal(sheet, shuffled)
    probability = sheet[sheet.sample_group == "stratified_probability"]
    challenge = sheet[sheet.sample_group == "challenge_nonprobability"]
    assert len(probability) == 3 and len(challenge) == 4
    assert set(probability["o:id"]).isdisjoint(challenge["o:id"])
    assert probability.design_weight.sum() == 12
    assert challenge.design_weight.isna().all()
    assert set(audit.sample_group) == set(sheet.sample_group)
    assert all(sheet.adjudicated_polarity.eq(""))
    assert not any(col.endswith("_polarite") for col in sheet)
    assert "text_excerpt" not in sheet
    assert summary["annotation_generation"] == models[0].generation


def test_expert_challenge_unknown_ids_rejected(corpus):
    with pytest.raises(ValueError, match="absent from this snapshot"):
        annotation_sample(corpus, difficult_ids=["9999"])


def test_annotation_excerpt_requires_explicit_option(corpus):
    sheet, _, _ = annotation_sample(corpus, per_stratum=1, challenge_size=0, excerpt_chars=7)
    assert sheet.text_excerpt.tolist() == ["private"]


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
    ("annotation_review", ["--per-stratum", "1", "--challenge-size", "0"], "annotation_review_sheet.csv"),
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
