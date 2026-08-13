"""Index authority aggregates — what `frequency` actually counts.

`frequency`/`first_occurrence`/`last_occurrence`/`countries` are the only
numbers in the dataset computed by scanning other subsets, and they are joined
on an exact `Titre` match against a controlled vocabulary. Both halves of that
are easy to get subtly wrong — a substring match inflates every short name, and
scanning several fields per row counts one document more than once — so the
semantics are pinned here rather than left to the shape of the loop.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = REPO_ROOT / "index" / "upload_index_hf.py"
    spec = importlib.util.spec_from_file_location("index_upload_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["index_upload_under_test"] = module
    spec.loader.exec_module(module)
    return module


idx = _load_module()


def _frame(rows):
    return pd.DataFrame(rows)


EMPTY = _frame([])


def _stats(**frames):
    """Run the aggregate over one named subset, the others empty."""
    return idx.calculate_frequency_stats(
        frames.get("articles", EMPTY),
        frames.get("publications", EMPTY),
        frames.get("references", EMPTY),
        frames.get("audiovisual", EMPTY),
    )


class TestFrequencyCountsItems:
    """One item contributes at most 1 to a term, however often it names it."""

    def test_a_term_in_two_scanned_fields_counts_once(self):
        # 'Burkina Faso' as both subject and spatial is one document, not two.
        # Before this rule, entities that happened to appear in several fields
        # outranked equally-cited ones purely on field count.
        stats = _stats(articles=_frame([
            {"subject": "Burkina Faso", "spatial": "Burkina Faso",
             "author": "", "pub_date": "2020-01-01", "country": "Burkina Faso"},
        ]))
        assert stats["Burkina Faso"]["frequency"] == 1

    def test_a_term_repeated_inside_one_field_counts_once(self):
        stats = _stats(articles=_frame([
            {"subject": "Tabaski|Tabaski", "spatial": "", "author": "",
             "pub_date": "2020-01-01", "country": "Togo"},
        ]))
        assert stats["Tabaski"]["frequency"] == 1

    def test_distinct_items_each_count(self):
        stats = _stats(articles=_frame([
            {"subject": "Tabaski", "spatial": "", "author": "",
             "pub_date": "2020-01-01", "country": "Togo"},
            {"subject": "Tabaski", "spatial": "", "author": "",
             "pub_date": "2021-01-01", "country": "Benin"},
        ]))
        assert stats["Tabaski"]["frequency"] == 2

    def test_terms_are_matched_whole_not_by_substring(self):
        # 'Niger' must not pick up 'Nigeria'; the values are authority titles.
        stats = _stats(articles=_frame([
            {"subject": "", "spatial": "Nigeria", "author": "",
             "pub_date": "2020-01-01", "country": "Nigeria"},
        ]))
        assert "Niger" not in stats
        assert stats["Nigeria"]["frequency"] == 1

    def test_whitespace_around_a_pipe_is_stripped(self):
        stats = _stats(articles=_frame([
            {"subject": "Tabaski | Ramadan", "spatial": "", "author": "",
             "pub_date": "2020-01-01", "country": "Togo"},
        ]))
        assert stats["Tabaski"]["frequency"] == 1
        assert stats["Ramadan"]["frequency"] == 1


class TestOccurrenceWindowAndCountries:

    def test_first_and_last_span_every_subset(self):
        stats = _stats(
            articles=_frame([{"subject": "COSIM", "spatial": "", "author": "",
                              "pub_date": "2005-06-01", "country": "Côte d'Ivoire"}]),
            audiovisual=_frame([{"subject": "COSIM", "spatial": "", "creator": "",
                                 "publisher": "", "pub_date": "2021-03-04",
                                 "country": "Côte d'Ivoire"}]),
        )
        assert stats["COSIM"]["frequency"] == 2
        assert stats["COSIM"]["first_occurrence"] == "2005-06-01"
        assert stats["COSIM"]["last_occurrence"] == "2021-03-04"

    def test_countries_are_sorted_and_deduplicated(self):
        stats = _stats(articles=_frame([
            {"subject": "Ramadan", "spatial": "", "author": "",
             "pub_date": "2020-01-01", "country": "Togo"},
            {"subject": "Ramadan", "spatial": "", "author": "",
             "pub_date": "2020-01-02", "country": "Benin"},
            {"subject": "Ramadan", "spatial": "", "author": "",
             "pub_date": "2020-01-03", "country": "Togo"},
        ]))
        assert stats["Ramadan"]["countries"] == "Benin|Togo"

    def test_an_undated_item_still_counts(self):
        # references carry bibliographic years, and some carry nothing at all;
        # a missing date must not drop the item from the frequency.
        stats = _stats(references=_frame([
            {"author": "Madore, Frédérick", "editor": "", "publisher": "",
             "pub_date": "", "country": ""},
        ]))
        assert stats["Madore, Frédérick"]["frequency"] == 1
        assert stats["Madore, Frédérick"]["first_occurrence"] == ""


class TestAudiovisualIsScanned:
    """Issue #13: the YouTube channel authorities were invisible in the index."""

    def test_audiovisual_is_a_declared_source(self):
        assert "audiovisual" in idx.FREQUENCY_SOURCE_FIELDS

    def test_it_scans_publisher_so_a_channel_gets_a_frequency(self):
        # The channel is a foaf:Organization — an index row in its own right —
        # linked as dcterms:publisher. Without publisher in the field list it
        # would sit at frequency 0 no matter how many videos it published.
        assert "publisher" in idx.FREQUENCY_SOURCE_FIELDS["audiovisual"]
        stats = _stats(audiovisual=_frame([
            {"subject": "", "spatial": "Burkina Faso", "creator": "",
             "publisher": "L'Autregard", "pub_date": "2026-01-09",
             "country": "Burkina Faso"},
            {"subject": "", "spatial": "Burkina Faso", "creator": "",
             "publisher": "L'Autregard", "pub_date": "2026-02-11",
             "country": "Burkina Faso"},
        ]))
        assert stats["L'Autregard"]["frequency"] == 2
        assert stats["L'Autregard"]["first_occurrence"] == "2026-01-09"
        assert stats["L'Autregard"]["countries"] == "Burkina Faso"

    def test_it_uses_the_audiovisual_column_names(self):
        # The subset has creator/publisher where articles have author/newspaper;
        # naming the wrong columns fails silently as an all-zero scan.
        assert idx.FREQUENCY_SOURCE_FIELDS["audiovisual"] == [
            "subject", "spatial", "creator", "publisher"
        ]

    def test_a_missing_frame_is_skipped_rather_than_treated_as_empty(self):
        # calculate_frequency_stats tolerates None for the optional subset, but
        # the loader is what guarantees it is never None in production: a load
        # failure raises there instead of silently zeroing the statistics.
        stats = idx.calculate_frequency_stats(EMPTY, EMPTY, EMPTY, None)
        assert stats == {}
