"""Free, key-less football data sources.

* football-data.co.uk  - match results + shots, corners, cards, xG (recent seasons)
                          and bookmaker odds for many European leagues, many seasons.
* Fantasy Premier League API - Premier League players (goals, xG, xA, minutes, form),
                          per-gameweek player history and the upcoming fixture list.

Everything is cached on disk under data/cache so the app works offline after the
first load and only re-downloads when you press "Refresh" (or the cache is stale).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STALE_AFTER = 6 * 3600  # seconds before the current season auto-refreshes

LEAGUES = {
    "E0": {"name": "Premier League", "country": "England", "players": True},
    "E1": {"name": "Championship", "country": "England", "players": False},
    "SP1": {"name": "La Liga", "country": "Spain", "players": False},
    "D1": {"name": "Bundesliga", "country": "Germany", "players": False},
    "I1": {"name": "Serie A", "country": "Italy", "players": False},
    "F1": {"name": "Ligue 1", "country": "France", "players": False},
    "N1": {"name": "Eredivisie", "country": "Netherlands", "players": False},
    "P1": {"name": "Primeira Liga", "country": "Portugal", "players": False},
}

FDUK = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FDUK_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
FPL = "https://fantasy.premierleague.com/api"

HEADERS = {"User-Agent": "Mozilla/5.0 (football-analytics-app)"}

# FPL short names -> football-data.co.uk names
FPL_TO_FDUK = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nott'm Forest",
    "Wolves": "Wolves",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Sheffield Utd": "Sheffield United",
    "Luton Town": "Luton",
}


# --------------------------------------------------------------------------- seasons
def current_season_code(today: dt.date | None = None) -> str:
    """'2627' for the 2026/27 season (seasons roll over in July)."""
    today = today or dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def previous_seasons(season: str, n: int) -> list[str]:
    start = int(season[:2])
    out = []
    for i in range(1, n + 1):
        s = (start - i) % 100
        out.append(f"{s:02d}{(s + 1) % 100:02d}")
    return out


def season_label(season: str) -> str:
    return f"20{season[:2]}/{season[2:]}"


# --------------------------------------------------------------------------- http + cache
def _get(url: str) -> bytes:
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


def _cached(name: str, url: str, force: bool, max_age: float | None) -> bytes:
    path = CACHE_DIR / name
    if path.exists() and not force:
        age = time.time() - path.stat().st_mtime
        if max_age is None or age < max_age:
            return path.read_bytes()
    try:
        data = _get(url)
    except Exception:
        if path.exists():  # network down -> serve stale cache
            return path.read_bytes()
        raise
    path.write_bytes(data)
    return data


def cache_timestamp(name: str) -> float | None:
    p = CACHE_DIR / name
    return p.stat().st_mtime if p.exists() else None


# --------------------------------------------------------------------------- matches
_COLS = {
    "HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag", "FTR": "result",
    "HTHG": "hthg", "HTAG": "htag", "HS": "hs", "AS": "as", "HST": "hst", "AST": "ast",
    "HC": "hc", "AC": "ac", "HF": "hf", "AF": "af", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
    "HxG": "hxg", "AxG": "axg",
    "AvgH": "odds_h", "AvgD": "odds_d", "AvgA": "odds_a",
    "Avg>2.5": "odds_o25", "Avg<2.5": "odds_u25",
    "B365H": "b365_h", "B365D": "b365_d", "B365A": "b365_a",
}


def _parse_matches(raw: bytes, div: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    keep = {k: v for k, v in _COLS.items() if k in df.columns}
    out = df[list(keep)].rename(columns=keep).copy()
    for c in _COLS.values():
        if c not in out.columns:
            out[c] = np.nan
    # older seasons used BbAv* for average odds
    for old, new in (("BbAvH", "odds_h"), ("BbAvD", "odds_d"), ("BbAvA", "odds_a")):
        if old in df.columns and out[new].isna().all():
            out[new] = df[old]
    for c in ("odds_h", "odds_d", "odds_a"):  # fall back to Bet365 when no average
        out[c] = out[c].fillna(out[f"b365_{c[-1]}"])
    out["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if "Time" in df.columns:
        out["time"] = df["Time"].astype(str)
    out["div"] = div
    out["season"] = season
    out = out.dropna(subset=["date", "hg", "ag"])
    out["hg"] = out["hg"].astype(int)
    out["ag"] = out["ag"].astype(int)
    return out.sort_values("date").reset_index(drop=True)


def load_season(div: str, season: str, force: bool = False) -> pd.DataFrame:
    is_current = season == current_season_code()
    raw = _cached(
        f"{div}_{season}.csv",
        FDUK.format(season=season, div=div),
        force=force and is_current,
        max_age=STALE_AFTER if is_current else None,
    )
    return _parse_matches(raw, div, season)


def load_history(div: str, season: str, n_prev: int = 3, force: bool = False) -> pd.DataFrame:
    frames = [load_season(div, season, force=force)]
    for s in previous_seasons(season, n_prev):
        try:
            frames.append(load_season(div, s))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def load_fixtures_csv(force: bool = False) -> pd.DataFrame:
    raw = _cached("fixtures.csv", FDUK_FIXTURES, force=force, max_age=STALE_AFTER)
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", on_bad_lines="skip")
    if df.empty:
        return df
    out = pd.DataFrame({
        "div": df["Div"], "home": df["HomeTeam"], "away": df["AwayTeam"],
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "time": df.get("Time", pd.Series([""] * len(df))).astype(str),
        "odds_h": df.get("AvgH"), "odds_d": df.get("AvgD"), "odds_a": df.get("AvgA"),
        "odds_o25": df.get("Avg>2.5"), "odds_u25": df.get("Avg<2.5"),
    })
    return out


# --------------------------------------------------------------------------- FPL
def fpl_bootstrap(force: bool = False) -> dict:
    return json.loads(_cached("fpl_bootstrap.json", f"{FPL}/bootstrap-static/", force, STALE_AFTER))


def fpl_fixtures(force: bool = False) -> list[dict]:
    return json.loads(_cached("fpl_fixtures.json", f"{FPL}/fixtures/", force, STALE_AFTER))


def fpl_player_history(player_id: int, force: bool = False) -> dict:
    return json.loads(_cached(
        f"fpl_player_{int(player_id)}.json", f"{FPL}/element-summary/{int(player_id)}/", force, STALE_AFTER
    ))


def fpl_team_names(bootstrap: dict, known: set[str] | None = None) -> dict[int, str]:
    """FPL team id -> football-data.co.uk team name."""
    out = {}
    for t in bootstrap["teams"]:
        name = FPL_TO_FDUK.get(t["name"], t["name"])
        if known and name not in known:
            # fuzzy fallback: first known team sharing the first word
            first = name.split()[0].lower()
            match = next((k for k in known if k.lower().startswith(first)), None)
            name = match or name
        out[t["id"]] = name
    return out
