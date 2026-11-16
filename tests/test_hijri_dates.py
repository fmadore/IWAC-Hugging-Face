import pytest

import calculate_hijri_dates as hijri
from calculate_hijri_dates import (
    HIJRI_COLUMNS,
    HIJRI_MONTHS,
    add_hijri_batch,
    date_status,
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

    @pytest.mark.parametrize("value", [
        "+199-01-01", "1_99-01-01", "١٩٩٩-٠١-٠١",
        "1999-01-01 extra", "1999-01-01T",
    ])
    def test_only_ascii_digits_in_exact_iso_form(self, value):
        # int() accepted all of these; they are not dates.
        assert parse_gregorian(value) is None

    @pytest.mark.parametrize("value", ["1999-01-01T00:00:00", "1999-01-01T10:20:30.5+01:00",
                                       "1999-01-01T10:20Z"])
    def test_iso_datetime_uses_its_date_part(self, value):
        assert parse_gregorian(value) == (1999, 1, 1)


class TestDateStatus:
    @pytest.mark.parametrize("value, status", [
        ("1999-04-17", "converted"),
        ("1999-04-17T00:00:00", "converted_datetime"),
        ("1900-01-01", "outside_tables"),
        ("2077-11-17", "outside_tables"),
        ("1924-08-01", "converted"),
        ("1999-02-30", "invalid_date"),
        ("1998-06", "imprecise"),
        ("+199-01-01", "imprecise"),
        ("", "empty"),
        (None, "empty"),
    ])
    def test_buckets(self, value, status):
        assert date_status(value) == status

    def test_report_files_out_of_table_dates_on_their_own_row(self):
        import io

        import pandas as pd
        from rich.console import Console

        buffer = io.StringIO()
        original = hijri.console
        hijri.console = Console(file=buffer, width=200)
        try:
            dates = ["1900-01-01", "1999-04-17", "1998", "1999-04-17T00:00:00"]
            out = add_hijri_batch({"pub_date": list(dates)})
            hijri.report(pd.DataFrame({"pub_date": dates, **{c: out[c] for c in HIJRI_COLUMNS}}), "t")
        finally:
            hijri.console = original
        lines = {line.split("│")[1].strip(): line.split("│")[2].strip()
                 for line in buffer.getvalue().splitlines() if line.count("│") >= 3}
        assert lines["Outside Umm al-Qura tables (1924-08-01..2077-11-16)"] == "1"
        assert lines["Imprecise date (no lunar day)"] == "1"
        assert lines["Converted to Hijri"] == "2"
        assert lines["of which ISO datetime (date part used)"] == "1"


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

    def test_outside_the_tables_is_none(self):
        assert to_hijri(1924, 7, 31) is None
        assert to_hijri(2077, 11, 17) is None

    @pytest.mark.parametrize("date, expected", [
        # Pinned stored values: the parser rewrite must not move any of them.
        ("1961-01-01", (1380, 7, 14)),
        ("1975-12-31", (1395, 12, 28)),
        ("1989-06-14", (1409, 11, 11)),
        ("1999-04-17", (1420, 1, 1)),
        ("2000-02-29", (1420, 11, 23)),
        ("2012-08-19", (1433, 10, 1)),
        ("2024-03-20", (1445, 9, 10)),
        ("2025-05-13", (1446, 11, 15)),
    ])
    def test_regression_on_stored_conversions(self, date, expected):
        out = add_hijri_batch({"pub_date": [date]})
        assert tuple(out[c][0] for c in HIJRI_COLUMNS) == expected


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


class TestTypedMap:
    def test_a_first_batch_without_dates_does_not_crash_the_run(self):
        """Without declared output types, datasets fixes a column as ``null``
        from an all-None first batch and fails on the next one."""
        from datasets import Dataset, Value

        from _common import map_with_progress

        ds = Dataset.from_dict({
            "o:id": ["1", "2", "3", "4"],
            "pub_date": ["1998", "", "1999-04-17", "2024-03-20"],
            "nb_pages": [1, None, 3, 4],
        })
        out = map_with_progress(
            ds, add_hijri_batch, batch_size=2,
            output_types={c: Value("int64") for c in HIJRI_COLUMNS},
        )
        assert out.features["hijri_year"].dtype == "int64"
        assert out["hijri_year"][:] == [None, None, 1420, 1445]
        # Untouched columns keep their type (no pandas round trip).
        assert out.features["nb_pages"].dtype == "int64"


class TestStageWiring:
    def _configure(self, monkeypatch, dataset, argv):
        import io

        from rich.console import Console

        pushed = []
        state = {"ds": dataset}
        monkeypatch.setattr(hijri, "console", Console(file=io.StringIO()))
        monkeypatch.setattr(hijri, "ensure_hf_token", lambda **kwargs: "offline-placeholder")
        monkeypatch.setattr(hijri, "load_hub_dataset", lambda *a, **k: state["ds"])
        monkeypatch.setattr(hijri, "push_dataset", lambda ds, **k: pushed.append((ds, k)) or True)
        monkeypatch.setattr("sys.argv", ["hijri", *argv])
        return pushed, state

    def test_unchanged_run_does_not_push_and_commit_names_the_version(self, monkeypatch):
        from datasets import Dataset

        ds = Dataset.from_dict({"o:id": ["1", "2"], "pub_date": ["1999-04-17", "1998"]})
        pushed, state = self._configure(monkeypatch, ds, ["--config", "articles", "-y"])
        assert hijri.main() == 0
        assert len(pushed) == 1
        assert f"hijridate {hijri.hijridate_version()}" in pushed[0][1]["commit_message"]
        state["ds"] = pushed[0][0]
        monkeypatch.setattr("sys.argv", ["hijri", "--config", "articles", "--update-mode", "all", "-y"])
        assert hijri.main() == 0
        assert len(pushed) == 1  # identical values: nothing to push

    def test_yes_alone_means_missing(self, monkeypatch):
        from datasets import Dataset

        ds = Dataset.from_dict({"o:id": ["1"], "pub_date": ["1999-04-17"],
                                "hijri_year": [1111], "hijri_month": [2], "hijri_day": [3]})
        pushed, _ = self._configure(monkeypatch, ds, ["--config", "articles", "-y"])
        assert hijri.main() == 0
        assert pushed == []  # complete row kept by 'missing': nothing changed
