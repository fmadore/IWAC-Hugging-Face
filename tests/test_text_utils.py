from iwac_common.text_utils import simple_tokenize, tokenize_words


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

    def test_elision_split(self):
        assert tokenize_words("l'islam") == ["islam"]
        assert tokenize_words("d'abord qu'il n'est") == ["abord", "il", "est"]

    def test_curly_apostrophe(self):
        assert tokenize_words("l’imam") == ["imam"]

    def test_aujourdhui_stays_one_token(self):
        assert tokenize_words("aujourd'hui") == ["aujourdhui"]
        assert tokenize_words("Aujourd’hui même") == ["aujourdhui", "même"]

    def test_compound_elisions(self):
        assert tokenize_words("jusqu'à lorsqu'on") == ["à", "on"]

    def test_digits_kept(self):
        assert tokenize_words("En 1995, 3 mosquées") == ["en", "1995", "3", "mosquées"]

    def test_empty(self):
        assert tokenize_words("") == []
        assert tokenize_words(None) == []
