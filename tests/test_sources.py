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
    def fake_get(url, headers=None):
        if "2728" in url:
            raise RuntimeError("404")
        return CSV, {}
    monkeypatch.setattr(sources, "_get", fake_get)
    assert sources.active_season("E0", dt.date(2027, 7, 10)) == "2627"


def test_season_with_no_matches_yet_falls_back(cache, monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda url, headers=None: (CSV.split(b"\n")[0] + b"\n" if "2728" in url else CSV, {}))
    assert sources.active_season("E0", dt.date(2027, 8, 1)) == "2627"


def test_finished_season_keeps_refreshing_until_downloaded_after_it_ended(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(sources, "_get", lambda url, headers=None: (calls.append(url) or CSV, {}))
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

    def boom(url, headers=None):
        raise RuntimeError("site down")
    monkeypatch.setattr(sources, "_get", boom)
    df = sources.load_season("E0", "2627", force=True)
    assert len(df) == 1
    assert "E0_2627.csv" in sources.STALE_SERVED


def test_error_page_is_not_saved_as_results(cache, monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda url, headers=None: (b"<html>Service unavailable</html>", {}))
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
    monkeypatch.setattr(sources, "_get", lambda url, headers=None: (fx, {}))
    df = sources.load_fixtures_csv()
    assert list(df.kickoff) == ["2026-11-21T20:00:00Z", "2026-08-15T19:00:00Z"]


def test_unchanged_file_is_not_downloaded_again(cache, monkeypatch):
    """Second request sends the saved validators; a 304 reuses the saved copy."""
    seen = []

    def fake_get(url, headers=None):
        seen.append(dict(headers or {}))
        if headers and headers.get("If-None-Match") == '"v1"':
            return None, {}                                   # 304 Not Modified
        return CSV, {"etag": '"v1"', "last-modified": "Mon, 09 Nov 2026 10:03:06 GMT"}
    monkeypatch.setattr(sources, "_get", fake_get)
    first = sources.load_season("E0", "2627", force=True)
    second = sources.load_season("E0", "2627", force=True)
    assert seen[0] == {}
    assert seen[1] == {"If-None-Match": '"v1"', "If-Modified-Since": "Mon, 09 Nov 2026 10:03:06 GMT"}
    assert len(first) == len(second) == 1
    assert "E0_2627.csv" not in sources.STALE_SERVED


def test_player_history_needs_one_request_per_new_gameweek(cache, monkeypatch):
    calls = []

    def live(gw):
        return {"elements": [{"id": 7, "stats": {"minutes": 90 - gw, "goals_scored": 1, "assists": 0,
                                                 "expected_goals": "0.5", "expected_assists": "0.1",
                                                 "total_points": 6},
                              "explain": [{"fixture": 100 + gw}]}]}

    def fake_get(url, headers=None):
        calls.append(url)
        gw = int(url.rstrip("/").split("/")[-2])
        return __import__("json").dumps(live(gw)).encode(), {}
    monkeypatch.setattr(sources, "_get", fake_get)
    boot = {"events": [{"id": g, "finished": g <= 3, "data_checked": g <= 3} for g in range(1, 6)],
            "elements": [{"id": 7, "team": 1}]}
    fixtures = [{"id": 100 + g, "team_h": 1 if g % 2 else 2, "team_a": 2 if g % 2 else 1} for g in range(1, 6)]
    h = sources.fpl_player_histories(boot, fixtures)
    assert [g["round"] for g in h[7]] == [1, 2, 3]            # unfinished gameweeks are skipped
    assert h[7][0]["opponents"] == [2] and h[7][0]["home"] is True
    assert h[7][1]["home"] is False
    assert len(calls) == 3
    boot["events"][3].update(finished=True, data_checked=True)  # a week later: gameweek 4 done
    sources.fpl_player_histories(boot, fixtures)
    assert len(calls) == 4                                    # only the new gameweek was fetched
