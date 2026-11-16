"""Outlet → country lookup for pipe-joined publisher values. No network."""

import asyncio
import importlib

import pytest

from country_mapper import get_country_from_newspaper, get_country_from_outlets
from iwac_common import omeka_client


def test_single_outlet_matches_the_exact_lookup():
    assert get_country_from_outlets("Sidwaya") == get_country_from_newspaper("Sidwaya")
    assert get_country_from_outlets("Sidwaya") == "Burkina Faso"


def test_two_publishers_resolve_through_the_first_known_one():
    # The exact lookup misses the joined string entirely.
    assert get_country_from_newspaper("Unknown Outlet|Le Sahel") == ""
    assert get_country_from_outlets("Unknown Outlet | Le Sahel") == "Niger"
    assert get_country_from_outlets("La Nation|Le Sahel") == "Benin"


@pytest.mark.parametrize("value", ["", None, "Unknown|Other"])
def test_unknown_or_empty_is_blank(value):
    assert get_country_from_outlets(value) == ""


class _Api:
    def _secrets(self):
        return ()


@pytest.mark.parametrize("subset", ["articles", "publications"])
def test_press_mappers_resolve_a_two_publisher_item(monkeypatch, subset):
    from iwac_pipeline.cli import UPLOAD_MODULES

    module = importlib.import_module(dict(UPLOAD_MODULES)[subset])

    async def no_thumbnail(*args, **kwargs):
        return ""

    monkeypatch.setattr(omeka_client, "fetch_iiif_thumbnail_url", no_thumbnail)
    item = {
        "o:id": 9, "o:is_public": True,
        "dcterms:publisher": [
            {"display_title": "Bulletin inconnu", "is_public": True},
            {"display_title": "Le Sahel", "is_public": True},
        ],
    }
    row = asyncio.run(module.SPEC.map_item(item, _Api()))
    assert row["newspaper"] == "Bulletin inconnu|Le Sahel"
    assert row["country"] == "Niger"
