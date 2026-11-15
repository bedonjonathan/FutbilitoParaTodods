"""Season roll-over, refresh rules, team-name matching and kick-off times."""
import datetime as dt
import os

import pytest

from backend import sources

CSV = b"Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,16/08/2026,15:00,A,B,1,0,H\n"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
    sources._active.clear()
    sources.STALE_SERVED.clear()
    return tmp_path


def test_season_codes():
    assert sources.current_season_code(dt.date(2026, 11, 15)) == "2627"
    assert sources.current_season_code(dt.date(2027, 5, 20)) == "2627"
    assert sources.current_season_code(dt.date(2027, 7, 1)) == "2728"
    assert sources.previous_seasons("2627", 2) == ["2526", "2425"]
    assert sources.season_label("2627") == "2026/27"


def test_july_falls_back_to_last_season(cache, monkeypatch):
    # in July the new season's file does not exist yet
    def fake_get(url):
        if "2728" in url:
            raise RuntimeError("404")
        return CSV
    monkeypatch.setattr(sources, "_get", fake_get)
    assert sources.active_season("E0", dt.date(2027, 7, 10)) == "2627"


def test_season_with_no_matches_yet_falls_back(cache, monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda url: CSV.split(b"\n")[0] + b"\n" if "2728" in url else CSV)
    assert sources.active_season("E0", dt.date(2027, 8, 1)) == "2627"


def test_finished_season_keeps_refreshing_until_downloaded_after_it_ended(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(sources, "_get", lambda url: calls.append(url) or CSV)
    f = cache / "E0_2526.csv"
    f.write_bytes(CSV)
    before_end = dt.datetime(2026, 5, 14).timestamp()      # saved before the final matchday
    os.utime(f, (before_end, before_end))
    assert not sources.season_is_final("E0", "2526")
    sources.load_season("E0", "2526")
    assert len(calls) == 1                                 # it was fetched again
    assert sources.season_is_final("E0", "2526")           # now saved after the season ended
    sources.load_season("E0", "2526", force=True)
    assert len(calls) == 1                                 # and is never fetched again


def test_failed_download_serves_saved_copy_and_flags_it(cache, monkeypatch):
    (cache / "E0_2627.csv").write_bytes(CSV)

    def boom(url):
        raise RuntimeError("site down")
    monkeypatch.setattr(sources, "_get", boom)
    df = sources.load_season("E0", "2627", force=True)
    assert len(df) == 1
    assert "E0_2627.csv" in sources.STALE_SERVED


def test_error_page_is_not_saved_as_results(cache, monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda url: b"<html>Service unavailable</html>")
    with pytest.raises(ValueError):
        sources.load_season("E0", "2627")
    assert not (cache / "E0_2627.csv").exists()


def test_team_matching_never_guesses():
    known = {"Hull", "Man United", "Man City", "Tottenham", "West Ham", "West Brom", "Newcastle",
             "Sheffield United", "Nott'm Forest", "Wolves"}
    assert sources.match_team("Hull City", known) == "Hull"
    assert sources.match_team("Man Utd", known) == "Man United"
    assert sources.match_team("Man City", known) == "Man City"
    assert sources.match_team("Spurs", known) == "Tottenham"
    assert sources.match_team("West Ham", known) == "West Ham"      # not West Brom
    assert sources.match_team("Sheffield Utd", known) == "Sheffield United"
    assert sources.match_team("Atlantis", known) is None
    names, unmatched = sources.fpl_team_names({"teams": [{"id": 1, "name": "Spurs"}, {"id": 2, "name": "Atlantis"}]}, known)
    assert names == {1: "Tottenham"} and unmatched == ["Atlantis"]


def test_kickoff_times_are_converted_from_uk_time(cache, monkeypatch):
    fx = (b"Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n"
          b"SP1,21/11/2026,20:00,A,B,2.0,3.4,3.8\n"      # winter: UK time = UTC
          b"SP1,15/08/2026,20:00,C,D,2.0,3.4,3.8\n")     # summer: UK time = UTC+1
    monkeypatch.setattr(sources, "_get", lambda url: fx)
    df = sources.load_fixtures_csv()
    assert list(df.kickoff) == ["2026-11-21T20:00:00Z", "2026-08-15T19:00:00Z"]
