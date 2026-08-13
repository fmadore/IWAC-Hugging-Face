"""Audiovisual mapping — the two populations sharing resource class 38.

Since 2026-08-12 class 38 holds embedded YouTube videos (template 23) beside
the deposited DVD/CD recordings (template 19), and they behave oppositely
wherever the mapping reaches for a media file. The fixtures below are trimmed
copies of one live item of each kind, so a regression on either shape fails
here rather than on the Hub.

The failure these guard against is silent by construction: every one of them
produces a well-formed row of the right dtype, with the row count unchanged,
so no upload rail fires.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = REPO_ROOT / "audiovisual" / "upload_audiovisual_hf.py"
    spec = importlib.util.spec_from_file_location("av_upload_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["av_upload_under_test"] = module
    spec.loader.exec_module(module)
    return module


av = _load_module()


# --------------------------------------------------------------------------
# Fixtures — trimmed from the live API (items 108263 and 15850)
# --------------------------------------------------------------------------

YOUTUBE_ITEM = {
    "o:id": 108263,
    "o:title": "FAIB: El Hadj Moussa KOUANDA succède à Imam Aboubacar YUGO.",
    "o:created": {"@value": "2026-08-12T11:24:14+00:00"},
    "o:resource_class": {"o:id": 38},
    "o:resource_template": {"o:id": 23},
    "o:primary_media": {"@id": "https://islam.zmo.de/api/media/108264"},
    "o:item_set": [{"o:id": 108260}],
    "thumbnail_display_urls": {
        "large": "https://islam.zmo.de/files/large/1e25.jpg",
        "medium": "https://islam.zmo.de/files/medium/1e25.jpg",
    },
    "dcterms:identifier": [{"@value": "iwac-video-0000048"}],
    "dcterms:title": [{"@value": "FAIB: El Hadj Moussa KOUANDA succède à Imam Aboubacar YUGO."}],
    "dcterms:publisher": [{"display_title": "L'Autregard", "value_resource_id": 108262}],
    "dcterms:description": [{"@value": "Cérémonie de passation de présidence de la FAIB."}],
    "dcterms:date": [{"@value": "2026-01-09"}],
    "dcterms:extent": [{"@value": "PT6M51S"}],
    "dcterms:type": [{"display_title": "Enregistrement vidéo"}],
    "dcterms:medium": [{"display_title": "Vidéo sur le web"}],
    "dcterms:spatial": [{"display_title": "Burkina Faso"}],
    "dcterms:rights": [
        {"@id": "http://rightsstatements.org/vocab/InC/1.0/", "o:label": "In Copyright"}
    ],
    "dcterms:language": [{"display_title": "Français"}, {"display_title": "Mooré"}],
    "dcterms:contributor": [{"display_title": "Frédérick Madore"}],
    "fabio:hasURL": [{"@id": "https://www.youtube.com/watch?v=xcGWG5msEEs"}],
    "bibo:content": [{"@value": "[00:00:00] Bonjour à tous.", "is_public": True}],
}

DEPOSITED_ITEM = {
    "o:id": 15850,
    "o:created": {"@value": "2023-03-01T09:00:00+00:00"},
    "o:resource_class": {"o:id": 38},
    "o:resource_template": {"o:id": 19},
    "o:primary_media": {"@id": "https://islam.zmo.de/api/media/15851"},
    "o:item_set": [{"o:id": 2184}],
    "dcterms:identifier": [{"@value": "iwac-audiovisual-0000001"}],
    "dcterms:title": [{"@value": "Sermon du vendredi"}],
    "dcterms:publisher": [{"display_title": "Daarul Hadeethis Salafiyyah"}],
    "dcterms:creator": [{"display_title": "Sheikh Abdullahi"}],
    "dcterms:date": [{"@value": "2015"}],
    "dcterms:extent": [{"@value": "PT45M"}],
    "dcterms:medium": [{"display_title": "DVD"}],
    "dcterms:type": [{"display_title": "Enregistrement vidéo"}],
    # A city sits alongside the country, and the country label is French.
    "dcterms:spatial": [{"display_title": "Nigéria"}, {"display_title": "Zaria"}],
    "dcterms:rights": [
        {
            "@id": "http://rightsstatements.org/vocab/InC-RUU/1.0/",
            "o:label": "In Copyright - Rights-Holder(s) Unlocatable or Unidentifiable",
        }
    ],
    "dcterms:language": [{"display_title": "Haoussa"}],
    "dcterms:contributor": [
        {"display_title": "Aleksei Akseshin"},
        {"display_title": "Vincent Favier"},
    ],
    "bibo:issue": [{"@value": "12"}],
}


class _FakeApi:
    """Stands in for OmekaApiClient: returns the media record by id.

    ``o:original_url`` is present-but-null on a YouTube media (the core
    ingester stores thumbnails only), which is exactly the shape that used to
    become the literal string "None".
    """

    MEDIA = {
        "108264": {"o:ingester": "youtube", "o:original_url": None, "o:media_type": None},
        "15851": {
            "o:ingester": "upload",
            "o:original_url": "https://islam.zmo.de/files/original/sermon.mp4",
            "o:media_type": "video/mp4",
        },
    }

    async def fetch_media_data(self, media_id):
        return self.MEDIA[media_id]


def _map(item, monkeypatch, *, thumbnail="https://islam.zmo.de/iiif-thumb.jpg"):
    """Run the async mapper with the two network helpers stubbed out."""

    async def fake_thumbnail(_omeka_id, _session):
        return thumbnail

    class _FakeConnManager:
        async def get(self):
            return None

    monkeypatch.setattr(av, "fetch_iiif_thumbnail_url", fake_thumbnail)
    monkeypatch.setattr(av, "conn_manager", _FakeConnManager())
    return asyncio.run(av.map_audiovisual_document(item, _FakeApi()))


# --------------------------------------------------------------------------
# Country resolution
# --------------------------------------------------------------------------

class TestCountry:
    """The subset-wide "Nigeria" default is gone; every row resolves on its own."""

    def test_youtube_item_resolves_burkina_faso_from_its_item_set(self):
        assert av._resolve_country(YOUTUBE_ITEM) == "Burkina Faso"

    def test_deposited_item_resolves_nigeria_from_the_french_place_label(self):
        # 'Nigéria' → 'Nigeria': the authority is French-labelled, the dataset's
        # country column is not.
        assert av._resolve_country(DEPOSITED_ITEM) == "Nigeria"

    def test_a_city_alongside_the_country_does_not_win(self):
        # 'Zaria' precedes nothing here, but the map must ignore non-country
        # places entirely rather than take the first spatial value.
        item = dict(DEPOSITED_ITEM, **{
            "dcterms:spatial": [{"display_title": "Zaria"}, {"display_title": "Nigéria"}]
        })
        assert av._resolve_country(item) == "Nigeria"

    def test_neighbouring_country_outside_the_collection_is_not_a_country(self):
        # Live item 78275 is spatial ['Bénin', 'Mali']; Mali is not one of the
        # six, so Benin must win regardless of order.
        item = {"dcterms:spatial": [{"display_title": "Mali"}, {"display_title": "Bénin"}]}
        assert av._resolve_country(item) == "Benin"

    def test_publisher_is_the_last_resort(self):
        # Live item 78278: no spatial, no country item set, Radio Oméga only.
        item = {"dcterms:publisher": [{"display_title": "Radio Oméga"}]}
        assert av._resolve_country(item) == "Burkina Faso"

    def test_format_item_sets_never_resolve_a_country(self):
        # 2183/2184 group by format, not place — putting them in the map would
        # stamp a country on recordings from anywhere.
        assert 2183 not in av.ITEM_SET_COUNTRY
        assert 2184 not in av.ITEM_SET_COUNTRY

    def test_unresolvable_item_is_blank_not_guessed(self):
        assert av._resolve_country({"o:id": 1}) == ""


# --------------------------------------------------------------------------
# Duration
# --------------------------------------------------------------------------

class TestDuration:
    """``extent`` stays verbatim; ``duration_seconds`` is the analysable form."""

    @pytest.mark.parametrize("iso,expected", [
        ("PT6M51S", 411),        # YouTube API precision
        ("PT45M", 2700),         # deposited-recording convention
        ("PT1H2M3S", 3723),
        ("PT30S", 30),
        ("P1DT1H", 90000),
        ("PT1.5S", 2),           # rounded, not truncated
    ])
    def test_parses(self, iso, expected):
        assert av._parse_iso_duration_seconds(iso) == expected

    @pytest.mark.parametrize("bad", ["", None, "6:51", "PT", "P", "quarante minutes"])
    def test_unparseable_is_none_not_zero(self, bad):
        # None and 0 are different claims: "no duration recorded" vs "empty video".
        assert av._parse_iso_duration_seconds(bad) is None


# --------------------------------------------------------------------------
# The mapped row
# --------------------------------------------------------------------------

class TestYoutubeRow:

    def test_media_pointers(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch)
        # No file behind the media: PDF must be empty, and emphatically not
        # the string "None" that str(null) produced.
        assert row["PDF"] == ""
        # The manifest resolves 200 with zero canvases — publishing it would
        # hand consumers a link that opens an empty viewer.
        assert row["iiif_manifest"] == ""
        # The thumbnail, by contrast, genuinely exists.
        assert row["thumbnail"] == "https://islam.zmo.de/iiif-thumb.jpg"

    def test_thumbnail_falls_back_to_the_baked_in_derivative(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch, thumbnail="")
        assert row["thumbnail"] == "https://islam.zmo.de/files/large/1e25.jpg"

    def test_canonical_watch_url(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch)
        assert row["URL"] == "https://www.youtube.com/watch?v=xcGWG5msEEs"

    def test_human_description_is_not_the_ai_summary(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch)
        assert row["description"].startswith("Cérémonie de passation")
        # bibo:shortDescription is absent on this item; descriptionAI must stay
        # empty rather than borrow the human blurb.
        assert row["descriptionAI"] == ""
        assert row["descriptionAI_en"] == ""

    def test_provenance_and_format_fields(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch)
        assert row["source_type"] == "youtube"
        assert row["type"] == "Enregistrement vidéo"
        assert row["medium"] == "Vidéo sur le web"
        assert row["rights"] == "In Copyright"
        assert row["contributor"] == "Frédérick Madore"
        assert row["publisher"] == "L'Autregard"
        assert row["country"] == "Burkina Faso"

    def test_duration_and_date(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch)
        assert row["extent"] == "PT6M51S"
        assert row["duration_seconds"] == 411
        # A complete YYYY-MM-DD, so calculate_hijri_dates.py can convert it.
        assert row["pub_date"] == "2026-01-09"

    def test_transcription_and_word_count(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch)
        assert row["OCR"].startswith("[00:00:00]")
        assert row["OCR_is_public"] is True
        assert row["nb_mots"] == 6  # 00 00 00 Bonjour à tous

    def test_multivalued_language_is_pipe_joined(self, monkeypatch):
        row = _map(YOUTUBE_ITEM, monkeypatch)
        assert row["language"] == "Français|Mooré"


class TestDepositedRow:
    """The legacy recordings must come through the change unchanged."""

    def test_file_pointers_survive(self, monkeypatch):
        row = _map(DEPOSITED_ITEM, monkeypatch)
        assert row["PDF"] == "https://islam.zmo.de/files/original/sermon.mp4"
        # A real file means a real manifest with canvases.
        assert row["iiif_manifest"] == "https://islam.zmo.de/iiif/3/15850/manifest"
        assert row["thumbnail"] == "https://islam.zmo.de/iiif-thumb.jpg"

    def test_metadata(self, monkeypatch):
        row = _map(DEPOSITED_ITEM, monkeypatch)
        assert row["source_type"] == "deposited"
        assert row["country"] == "Nigeria"
        assert row["medium"] == "DVD"
        assert row["creator"] == "Sheikh Abdullahi"
        assert row["issue"] == "12"
        assert row["duration_seconds"] == 2700
        assert row["contributor"] == "Aleksei Akseshin|Vincent Favier"

    def test_no_transcription_means_zero_words_not_a_crash(self, monkeypatch):
        row = _map(DEPOSITED_ITEM, monkeypatch)
        assert row["OCR"] == ""
        assert row["nb_mots"] == 0
        assert row["OCR_is_public"] is False

    def test_absent_url_is_empty(self, monkeypatch):
        row = _map(DEPOSITED_ITEM, monkeypatch)
        assert row["URL"] == ""


class TestBothShapesAgreeOnColumns:

    def test_same_columns_regardless_of_population(self, monkeypatch):
        # A per-population column set would give the parquet writer a ragged
        # frame and break the dataset card's feature list.
        assert set(_map(YOUTUBE_ITEM, monkeypatch)) == set(_map(DEPOSITED_ITEM, monkeypatch))

    def test_every_mapped_column_is_publicly_allowlisted(self, monkeypatch):
        # publish_public.py aborts on any column missing from the allowlist;
        # catching that here beats catching it mid-publish.
        from iwac_common.repos import load_public_columns

        allowed = set(load_public_columns()["audiovisual"])
        assert set(_map(YOUTUBE_ITEM, monkeypatch)) <= allowed


class TestSourceType:

    def test_template_is_authoritative(self):
        assert av._source_type({"o:resource_template": {"o:id": 23}}) == "youtube"
        assert av._source_type({"o:resource_template": {"o:id": 19}}) == "deposited"

    def test_url_is_the_fallback_for_an_untemplated_item(self):
        assert av._source_type(
            {"fabio:hasURL": [{"@id": "https://youtu.be/xcGWG5msEEs"}]}
        ) == "youtube"

    def test_a_deposited_item_with_an_archive_link_stays_deposited(self):
        # Live item 78275 links to web.archive.org — a source link, not a video
        # platform, and it must not flip the population.
        assert av._source_type(
            {"fabio:hasURL": [{"@id": "https://web.archive.org/web/2020/https://x.net/a"}]}
        ) == "deposited"
