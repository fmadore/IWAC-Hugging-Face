import pytest

from iwac_common.text_utils import (
    TOKENIZER_VERSION,
    applies_french_elision,
    count_words,
    simple_tokenize,
    tokenize_words,
)


class TestSimpleTokenize:
    def test_lowercases_and_splits(self):
        assert simple_tokenize("Le Grand IMAM") == ["le", "grand", "imam"]

    def test_min_len_drops_single_chars(self):
        assert simple_tokenize("a bb c dd") == ["bb", "dd"]

    def test_stopwords_checked_after_lowercasing(self):
        # The keyness bug: uppercase tokens evaded lowercase stopword sets.
        assert simple_tokenize("ISLAM paix", {"islam"}) == ["paix"]

    def test_empty_and_none_like(self):
        assert simple_tokenize("") == []
        assert simple_tokenize(None) == []

    def test_custom_min_len(self):
        assert simple_tokenize("ab abc", min_len=3) == ["abc"]


class TestTokenizeWords:
    def test_plain_words(self):
        assert tokenize_words("La mosquée de Ouaga") == ["la", "mosquée", "de", "ouaga"]

    def test_elided_clitic_is_its_own_token(self):
        assert tokenize_words("l'islam") == ["l'", "islam"]
        assert tokenize_words("d'abord qu'il n'est") == ["d'", "abord", "qu'", "il", "n'", "est"]

    def test_curly_apostrophe_is_normalised(self):
        assert tokenize_words("l’imam") == ["l'", "imam"]
        assert tokenize_words("Jama’at", language="Haoussa") == ["jama'at"]

    def test_aujourdhui_and_quelquun_stay_one_token(self):
        assert tokenize_words("aujourd'hui") == ["aujourd'hui"]
        assert tokenize_words("Aujourd’hui même") == ["aujourd'hui", "même"]
        assert tokenize_words("quelqu'un") == ["quelqu'un"]

    def test_compound_elisions(self):
        assert tokenize_words("jusqu'à lorsqu'on") == ["jusqu'", "à", "lorsqu'", "on"]

    @pytest.mark.parametrize("language", ["Français", "Anglais", "Haoussa", None])
    def test_transliterations_are_one_token_in_every_language(self, language):
        assert tokenize_words("Jama'at da'wa Qur'an", language=language) == ["jama'at", "da'wa", "qur'an"]

    def test_non_french_rows_keep_apostrophes_inside_words(self):
        assert tokenize_words("l'islam d'amour", language="Anglais") == ["l'islam", "d'amour"]
        assert tokenize_words("l'islam", language=None) == ["l'islam"]

    def test_edge_apostrophes_are_punctuation(self):
        assert tokenize_words("'Ulama' l' islam", language="Anglais") == ["ulama", "l", "islam"]

    def test_digits_kept(self):
        assert tokenize_words("En 1995, 3 mosquées") == ["en", "1995", "3", "mosquées"]

    def test_empty(self):
        assert tokenize_words("") == []
        assert tokenize_words(None) == []


class TestLanguageRule:
    @pytest.mark.parametrize("value, expected", [
        ("Français", True), (" Anglais | Français ", True), (["Français"], True),
        ("Anglais", False), ("Ewé", False), ("", False), (None, False),
    ])
    def test_elision_applies_when_the_languages_include_french(self, value, expected):
        assert applies_french_elision(value) is expected


class TestCountWords:
    def test_elided_clitic_counts_like_an_article(self):
        assert count_words("l'église") == count_words("la mosquée") == 2

    def test_language_is_passed_through(self):
        assert count_words("l'église", "Français|Anglais") == 2
        assert count_words("l'église", "Haoussa") == 1

    def test_backward_compatible_default_is_french(self):
        # The references/audiovisual upload mappers call count_words(text).
        assert count_words("qu'il") == count_words("qu'il", "Français") == 2

    def test_empty(self):
        assert count_words(None) == 0
        assert count_words("", "Français") == 0

    def test_version_bumped_for_the_new_rule(self):
        assert TOKENIZER_VERSION >= 3


def test_capital_n_m_names_count_as_one_word():
    from iwac_common.text_utils import tokenize_words

    assert tokenize_words("N'Guessan n'est pas à N'Djamena", language="Français") == [
        "n'guessan", "n'", "est", "pas", "à", "n'djamena",
    ]
