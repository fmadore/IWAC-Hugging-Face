"""All seven upload mappers on one synthetic item: shared helpers, new
analysable columns, and the public allowlist contract. No network."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from iwac_common import omeka_client
from iwac_common.field_mappers import (
    countries_from_item_sets,
    get_display_titles,
    get_literal_values,
    get_resource_ids,
    parse_coordinates,
    parse_pub_date,
    split_by_language,
)
from iwac_common.repos import load_public_columns
from iwac_pipeline.cli import UPLOAD_MODULES

ITEM = {
    "o:is_public": True,
    "o:id": 501,
    "o:title": "Un titre",
    "o:created": {"@value": "2024-03-14T10:00:00+00:00"},
    "o:resource_class": {"o:id": 36},
    "o:item_set": [{"o:id": 2192}, {"o:id": 23452}, {"o:id": 2193}, {"o:id": 2225}],
    "dcterms:identifier": [{"@value": "iwac-x-0000001"}],
    "dcterms:title": [{"@value": "Un titre"}],
    "dcterms:creator": [
        {"display_title": "Auteur A", "value_resource_id": 11},
        {"@value": "Auteur libre"},
    ],
    "dcterms:publisher": [{"display_title": "Sidwaya", "value_resource_id": 12}],
    "dcterms:date": [{"@value": "1998-06-14"}],
    "dcterms:subject": [
        {"display_title": "Ramadan", "value_resource_id": 21},
        {"display_title": "Tabaski", "value_resource_id": 22},
    ],
    "dcterms:spatial": [{"display_title": "Ouagadougou", "value_resource_id": 31}],
    "dcterms:language": [{"display_title": "Français"}],
    "dcterms:abstract": [
        {"@value": "Résumé.", "@language": "fr"},
        {"@value": "Summary.", "@language": "en"},
    ],
    "bibo:authorList": [{"display_title": "Auteur A", "value_resource_id": 11}],
    "bibo:editorList": [{"display_title": "Éditeur", "value_resource_id": 13}],
    "bibo:numPages": [{"@value": "iv, 250"}],
    "curation:coordinates": [{"@value": "12.37, -1.52"}],
    "bibo:content": [{"@value": "Le texte intégral de l'article.", "is_public": True}],
    "fabio:hasURL": [{"@id": "https://example.org/a"}],
}

SCRIPT_TO_SUBSET = dict(UPLOAD_MODULES)


class _Api:
    def _secrets(self):
        return ()


@pytest.fixture
def mapped(monkeypatch):
    monkeypatch.setenv("OMEKA_BASE_URL", "https://staging.example/api")
    monkeypatch.delenv("IWAC_PUBLIC_BASE_URL", raising=False)

    async def no_session():
        return None

    monkeypatch.setattr(omeka_client.conn_manager, "get", no_session)

    def run(subset):
        module = importlib.import_module(SCRIPT_TO_SUBSET[subset])

        async def no_thumbnail(*args, **kwargs):
            return ""

        monkeypatch.setattr(module, "fetch_iiif_thumbnail_url", no_thumbnail, raising=False)
        return asyncio.run(module.SPEC.map_item(dict(ITEM), _Api()))

    return run


@pytest.mark.parametrize("subset", list(UPLOAD_MODULES))
def test_every_mapped_column_is_publicly_allowlisted(mapped, subset):
    row = mapped(subset)
    assert set(row) <= load_public_columns()[subset]


@pytest.mark.parametrize("subset", ["articles", "publications", "documents", "images", "audiovisual"])
def test_primary_media_failure_preserves_all_dependent_pointers(mapped, monkeypatch, subset):
    monkeypatch.setitem(ITEM, "o:primary_media", {"@id": "https://example.test/api/media/1"})
    omeka_client.media_stats.reset()
    mapped(subset)  # _Api intentionally cannot retrieve primary media.
    pointer = "image_url" if subset == "images" else "PDF"
    assert omeka_client.media_stats.failed_fields_by_item["501"] == {
        pointer, "thumbnail", "iiif_manifest",
    }


@pytest.mark.parametrize("subset", list(UPLOAD_MODULES))
def test_item_urls_follow_the_configured_host(mapped, subset):
    assert mapped(subset)["iwac_url"] == (
        "https://staging.example/s/afrique_ouest/item/501"
    )


@pytest.mark.parametrize(
    "subset, expected",
    [
        ("articles", {"author_ids": "11", "newspaper_ids": "12",
                      "subject_ids": "21|22", "spatial_ids": "31"}),
        ("publications", {"author_ids": "11", "newspaper_ids": "12"}),
        ("documents", {"author_ids": "11", "subject_ids": "21|22"}),
        ("references", {"author_ids": "11", "editor_ids": "13",
                        "publisher_ids": "12", "spatial_ids": "31"}),
        ("audiovisual", {"creator_ids": "11", "publisher_ids": "12"}),
        ("images", {"creator_ids": "11", "subject_ids": "21|22"}),
    ],
)
def test_authority_ids_sit_beside_the_labels(mapped, subset, expected):
    row = mapped(subset)
    for column, value in expected.items():
        assert row[column] == value


@pytest.mark.parametrize("subset", [s for s in UPLOAD_MODULES if s != "index"])
def test_publication_year_and_precision(mapped, subset):
    row = mapped(subset)
    assert (row["pub_year"], row["pub_date_precision"]) == (1998, "day")


def test_references_abstract_is_split_by_language(mapped):
    row = mapped("references")
    assert (row["abstract"], row["abstract_en"]) == ("Résumé.", "Summary.")
    # Bibliographic numbers keep their verbatim form ("iv, 250" was blanked).
    assert row["nb_pages"] == "iv, 250"
    assert row["country"] == "Benin|Nigeria"
    assert row["URL"] == "https://example.org/a"


def test_country_item_sets_come_from_the_registry(mapped):
    assert mapped("documents")["country"] == "Benin"
    assert mapped("images")["country"] == "Benin"


@pytest.mark.parametrize("subset", ["images", "index"])
def test_coordinates_are_parsed(mapped, subset):
    row = mapped(subset)
    assert (row["latitude"], row["longitude"]) == (12.37, -1.52)


class TestHelpers:
    def test_resource_ids_skip_literals_and_duplicates(self):
        item = {"f": [{"value_resource_id": 5}, {"@value": "x"}, {"value_resource_id": 5}]}
        assert get_resource_ids(item, "f") == "5"

    def test_display_titles_ignore_literals_and_null_titles(self):
        item = {"f": [{"display_title": "A"}, {"display_title": None}, {"@value": "x"}, "junk"]}
        assert get_display_titles(item, "f") == "A"

    def test_literal_values(self):
        item = {"f": {"@value": "PT45M"}}
        assert get_literal_values(item, "f") == "PT45M"

    @pytest.mark.parametrize(
        "values, expected",
        [
            ([("Résumé", "fr"), ("Summary", "en")], ("Résumé", "Summary")),
            ([("Summary", "en")], ("Summary", "Summary")),
            ([("Un", None), ("Deux", None)], ("Un|Deux", "")),
            ([], ("", "")),
        ],
    )
    def test_split_by_language(self, values, expected):
        item = {"f": [
            {"@value": text, **({"@language": lang} if lang else {})}
            for text, lang in values
        ]}
        assert split_by_language(item, "f", "en") == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1998-06-14", (1998, "day")),
            ("1998-06", (1998, "month")),
            ("1998", (1998, "year")),
            ("1981-04/1981-06", (1981, "range")),
            ("1998-13", (None, "other")),
            ("s.d.", (None, "other")),
            ("", (None, "")),
            (None, (None, "")),
        ],
    )
    def test_parse_pub_date(self, raw, expected):
        assert parse_pub_date(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("12.37, -1.52", (12.37, -1.52)),
            ("12.37,-1.52", (12.37, -1.52)),
            ("95, 10", (None, None)),
            ("1, 2|3, 4", (None, None)),
            ("abc", (None, None)),
            ("", (None, None)),
        ],
    )
    def test_parse_coordinates(self, raw, expected):
        assert parse_coordinates(raw) == expected

    def test_country_item_sets(self):
        item = {"o:item_set": [{"o:id": 1}, {"o:id": 2}, {"o:id": 1}]}
        mapping = {1: "Benin", 2: "Togo"}
        assert countries_from_item_sets(item, mapping) == "Benin"
        assert countries_from_item_sets(item, mapping, first_only=False) == "Benin|Togo"

    def test_public_base_url_precedence(self, monkeypatch):
        monkeypatch.setenv("OMEKA_BASE_URL", "https://a.example/api/")
        monkeypatch.delenv("IWAC_PUBLIC_BASE_URL", raising=False)
        assert omeka_client.public_base_url() == "https://a.example"
        monkeypatch.setenv("IWAC_PUBLIC_BASE_URL", "https://b.example/")
        assert omeka_client.iiif_manifest_url(7) == "https://b.example/iiif/3/7/manifest"
        monkeypatch.delenv("IWAC_PUBLIC_BASE_URL")
        monkeypatch.delenv("OMEKA_BASE_URL")
        assert omeka_client.item_page_url(7) == (
            "https://islam.zmo.de/s/afrique_ouest/item/7"
        )


def test_unmapped_outlets_are_reported(capsys):
    import pandas as pd

    from iwac_common.upload_runner import report_unmapped_values

    df = pd.DataFrame({"newspaper": ["Sidwaya", "Nouveau Journal", ""],
                       "country": ["Burkina Faso", "", ""]})
    out = asyncio.run(report_unmapped_values("newspaper", "country")(df, None, "r", None))
    assert out is df
    assert "Nouveau Journal" in capsys.readouterr().out
