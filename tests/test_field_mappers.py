"""Language-aware value extraction — the guard against `"résumé|summary"`.

`bibo:shortDescription` carries one `fr` and one `en` literal per item since the
summariser went bilingual. `get_value()` would pipe-join them into a single
string, which fails quietly: no error, no row-count change, so none of the
upload rails fire. These tests pin the three value shapes the property actually
takes, plus the ordering assumption that made the pipe-join look safe.
"""

from iwac_common.field_mappers import get_value, get_value_by_language

FIELD = "bibo:shortDescription"


def _item(*values, oid=1):
    return {"o:id": oid, FIELD: list(values)}


def _lit(text, lang=None):
    v = {"@value": text}
    if lang:
        v["@language"] = lang
    return v


class TestBilingualPair:
    """The shape written by the current summariser: fr + en on one property."""

    ITEM = _item(_lit("Le résumé.", "fr"), _lit("The summary.", "en"))

    def test_french_column(self):
        assert get_value_by_language(self.ITEM, FIELD, "fr",
                                     untagged_matches=True) == "Le résumé."

    def test_english_column(self):
        assert get_value_by_language(self.ITEM, FIELD, "en") == "The summary."

    def test_get_value_would_have_pipe_joined_them(self):
        # Documents precisely what this function exists to avoid.
        assert get_value(self.ITEM, FIELD) == "Le résumé.|The summary."

    def test_order_in_the_payload_does_not_matter(self):
        # Omeka returns ('fr', 'en') today, but never promised to. The
        # pipe-join silently swapped languages if it ever flipped; keying off
        # @language means the reversed payload maps identically.
        reversed_item = _item(_lit("The summary.", "en"), _lit("Le résumé.", "fr"))
        assert get_value_by_language(reversed_item, FIELD, "fr",
                                     untagged_matches=True) == "Le résumé."
        assert get_value_by_language(reversed_item, FIELD, "en") == "The summary."


class TestUntaggedLegacy:
    """Summaries written before the pipeline went bilingual carry no tag."""

    ITEM = _item(_lit("Ancien résumé sans tag."))

    def test_counts_as_french(self):
        assert get_value_by_language(self.ITEM, FIELD, "fr",
                                     untagged_matches=True) == "Ancien résumé sans tag."

    def test_does_not_leak_into_english(self):
        # The 51 non-French/English articles keep their untagged French
        # summary; filing it under `descriptionAI_en` would be a lie.
        assert get_value_by_language(self.ITEM, FIELD, "en") == ""

    def test_untagged_ignored_when_flag_is_off(self):
        assert get_value_by_language(self.ITEM, FIELD, "fr") == ""


class TestPartialAndMissing:
    def test_french_only_leaves_english_empty(self):
        # The summariser reports a missing English counterpart rather than
        # failing, so this shape exists in the wild.
        item = _item(_lit("Résumé seul.", "fr"))
        assert get_value_by_language(item, FIELD, "fr", untagged_matches=True) == "Résumé seul."
        assert get_value_by_language(item, FIELD, "en") == ""

    def test_english_only_leaves_french_empty(self):
        item = _item(_lit("Summary only.", "en"))
        assert get_value_by_language(item, FIELD, "fr", untagged_matches=True) == ""
        assert get_value_by_language(item, FIELD, "en") == "Summary only."

    def test_absent_field(self):
        assert get_value_by_language({"o:id": 1}, FIELD, "fr") == ""

    def test_null_field(self):
        assert get_value_by_language({"o:id": 1, FIELD: None}, FIELD, "fr") == ""

    def test_empty_list(self):
        assert get_value_by_language(_item(), FIELD, "fr") == ""

    def test_bare_dict_is_accepted(self):
        item = {"o:id": 1, FIELD: _lit("Un seul.", "fr")}
        assert get_value_by_language(item, FIELD, "fr") == "Un seul."

    def test_blank_value_does_not_match(self):
        item = _item(_lit("", "fr"), _lit("The summary.", "en"))
        assert get_value_by_language(item, FIELD, "fr", untagged_matches=True) == ""
        assert get_value_by_language(item, FIELD, "en") == "The summary."


class TestDuplicateSameLanguage:
    def test_keeps_first_and_warns(self, caplog):
        item = _item(_lit("Premier.", "fr"), _lit("Second.", "fr"))
        with caplog.at_level("WARNING"):
            assert get_value_by_language(item, FIELD, "fr") == "Premier."
        # Must not silently re-create the pipe-joined shape.
        assert "keeping the first" in caplog.text


class TestFallback:
    """`index`'s Description: any value beats none, but only when asked."""

    def test_falls_back_to_other_language(self):
        item = _item(_lit("Eine Beschreibung.", "de"))
        assert get_value_by_language(item, "bibo:shortDescription", "fr",
                                     fallback=True) == "Eine Beschreibung."

    def test_no_fallback_by_default(self):
        item = _item(_lit("Eine Beschreibung.", "de"))
        assert get_value_by_language(item, FIELD, "fr") == ""

    def test_preferred_language_still_wins_over_fallback(self):
        item = _item(_lit("Eine Beschreibung.", "de"), _lit("Une description.", "fr"))
        assert get_value_by_language(item, FIELD, "fr", fallback=True) == "Une description."

    def test_fallback_reaches_untagged_value(self):
        # Reproduces the precedence the index-local helper had before the move:
        # tagged 'fr' first, then the first value of any kind.
        item = _item(_lit("Sans tag."))
        assert get_value_by_language(item, FIELD, "fr", fallback=True) == "Sans tag."
