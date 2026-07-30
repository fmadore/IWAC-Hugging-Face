import pytest

from calculate_hijri_dates import (
    HIJRI_COLUMNS,
    HIJRI_MONTHS,
    add_hijri_batch,
    parse_gregorian,
    to_hijri,
)


class TestParseGregorian:
    def test_complete_iso_date(self):
        assert parse_gregorian("1998-06-14") == (1998, 6, 14)

    @pytest.mark.parametrize("value", ["1998", "1998-06", "1981-04/1981-06", "", "  ", None])
    def test_imprecise_values_have_no_lunar_day(self, value):
        # A lunar month straddles two Gregorian ones, so a missing day is not a
        # rounding problem — it is unanswerable, and must stay None.
        assert parse_gregorian(value) is None

    def test_rejects_out_of_range_components(self):
        assert parse_gregorian("1998-13-01") is None
        assert parse_gregorian("1998-06-45") is None

    def test_rejects_wrong_separators_of_the_right_length(self):
        assert parse_gregorian("1998/06/14") is None

    def test_tolerates_surrounding_whitespace(self):
        assert parse_gregorian("  1998-06-14  ") == (1998, 6, 14)


class TestToHijri:
    def test_known_conversion(self):
        # 1 Muharram 1420 AH = 17 April 1999 (Umm al-Qura).
        assert to_hijri(1999, 4, 17) == (1420, 1, 1)

    def test_ramadan_is_month_nine(self):
        year, month, _ = to_hijri(2024, 3, 20)
        assert (year, month) == (1445, 9)
        assert HIJRI_MONTHS[month - 1] == "Ramadan"

    def test_collection_range_is_inside_the_tables(self):
        # The collection runs 1961-2025; both ends must convert.
        assert to_hijri(1961, 1, 1) is not None
        assert to_hijri(2025, 5, 13) is not None


class TestAddHijriBatch:
    def test_fills_three_columns(self):
        # 1999-04-17 = 1 Muharram 1420; 2024-03-20 = 10 Ramadan 1445
        # (Ramadan 1445 began on 11 March 2024).
        batch = {"pub_date": ["1999-04-17", "2024-03-20"]}
        out = add_hijri_batch(batch)
        assert [out[c] for c in HIJRI_COLUMNS] == [[1420, 1445], [1, 9], [1, 10]]

    def test_imprecise_rows_become_none_not_zero(self):
        # A 0 would be plotted as a real lunar month by any chart downstream.
        out = add_hijri_batch({"pub_date": ["1998", "1981-04/1981-06", None]})
        for col in HIJRI_COLUMNS:
            assert out[col] == [None, None, None]

    def test_missing_source_column_still_yields_aligned_columns(self):
        out = add_hijri_batch({"title": ["a", "b", "c"]})
        for col in HIJRI_COLUMNS:
            assert out[col] == [None, None, None]

    def test_empty_batch_does_not_raise(self):
        assert add_hijri_batch({}) == {c: [] for c in HIJRI_COLUMNS}

    def test_update_mode_missing_keeps_existing_values(self):
        batch = {
            "pub_date": ["1999-04-17", "2024-03-20"],
            "hijri_year": [1111, None],
            "hijri_month": [2, None],
            "hijri_day": [3, None],
        }
        out = add_hijri_batch(batch, update_mode="missing")
        assert out["hijri_year"] == [1111, 1445]  # row 0 preserved, row 1 computed
        assert out["hijri_month"] == [2, 9]

    def test_update_mode_all_overwrites(self):
        batch = {
            "pub_date": ["1999-04-17"],
            "hijri_year": [1111], "hijri_month": [2], "hijri_day": [3],
        }
        out = add_hijri_batch(batch, update_mode="all")
        assert out["hijri_year"] == [1420]


class TestMonthTable:
    def test_twelve_months_matching_the_module_used_by_the_website(self):
        # Same spellings as IwacVisualizations' shared/hijri.js, so a reader
        # moving between a chart and the dataset meets one transliteration.
        assert len(HIJRI_MONTHS) == 12
        assert HIJRI_MONTHS[8] == "Ramadan"
        assert HIJRI_MONTHS[11] == "Dhu al-Hijja"
