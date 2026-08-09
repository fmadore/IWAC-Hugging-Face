import asyncio

import pytest
from datasets import Dataset

import calculate_word_count as wc
from iwac_common.omeka_client import Config


def test_reference_content_batch_fails_if_one_item_is_unreadable(monkeypatch):
    client = wc.ReferenceContentClient(Config(), use_cache=False)

    async def fetch_item(item_id):
        if item_id == 2:
            raise OSError("offline")
        return {"bibo:content": [{"@value": "two words"}]}

    monkeypatch.setattr(client, "fetch_item", fetch_item)
    with pytest.raises(RuntimeError, match="existing word counts with zero"):
        asyncio.run(client.fetch_items_content([1, 2]))


def test_missing_mode_preserves_existing_reference_counts(monkeypatch):
    async def fetch_contents(self, item_ids):
        assert item_ids == [2]
        return {2: "two new words"}

    monkeypatch.setattr(
        wc.ReferenceContentClient, "fetch_items_content", fetch_contents
    )
    ds = Dataset.from_dict({"o:id": ["1", "2"], "nb_mots": [7, None]})
    result = asyncio.run(wc.process_references_word_count(
        ds,
        Config(),
        "nb_mots",
        update_mode="missing",
    ))
    assert result["nb_mots"] == [7, 3]
