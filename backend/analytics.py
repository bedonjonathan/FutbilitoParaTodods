"""League tables, team form series, Elo ratings and pre-match feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- long format
def team_matches(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (the home and away perspective stacked)."""
    cols = dict(
        date=df["date"], season=df["season"], match_idx=df.index,
    )
    home = pd.DataFrame({
        **cols, "team": df["home"], "opp": df["away"], "venue": "H",
        "gf": df["hg"], "ga": df["ag"], "xgf": df["hxg"], "xga": df["axg"],
        "sf": df["hs"], "sa": df["as"], "sotf": df["hst"], "sota": df["ast"],
        "cf": df["hc"], "ca": df["ac"],
    })
    away = pd.DataFrame({
        **cols, "team": df["away"], "opp": df["home"], "venue": "A",
        "gf": df["ag"], "ga": df["hg"], "xgf": df["axg"], "xga": df["hxg"],
        "sf": df["as"], "sa": df["hs"], "sotf": df["ast"], "sota": df["hst"],
        "cf": df["ac"], "ca": df["hc"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    long["res"] = np.select([long.gf > long.ga, long.gf == long.ga], ["W", "D"], "L")
    return long.sort_values(["date", "match_idx", "venue"], ascending=[True, True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- tables
def league_table(season_df: pd.DataFrame, venue: str | None = None) -> list[dict]:
    """Standings. venue='H' / 'A' gives the home-only / away-only tables from the sheet."""
    long = team_matches(season_df)
    if venue:
        long = long[long.venue == venue]
    g = long.groupby("team")
    t = pd.DataFrame({
        "M": g.size(), "W": g.res.apply(lambda s: (s == "W").sum()),
        "D": g.res.apply(lambda s: (s == "D").sum()), "L": g.res.apply(lambda s: (s == "L").sum()),
        "GF": g.gf.sum(), "GA": g.ga.sum(), "PTS": g.pts.sum(),
        "xGF": g.xgf.sum(min_count=1), "xGA": g.xga.sum(min_count=1),
        "SOTF": g.sotf.sum(min_count=1), "SOTA": g.sota.sum(min_count=1),
    })
    t["GD"] = t.GF - t.GA
    t["GF_pg"] = t.GF / t.M
    t["GA_pg"] = t.GA / t.M
    t["PPG"] = t.PTS / t.M
    t["xGD"] = t.xGF - t.xGA
    # form string: last 5 results, most recent last
    t["form"] = g.res.apply(lambda s: "".join(s.tail(5)))
    t = t.sort_values(["PTS", "GD", "GF"], ascending=False).reset_index()
    t.insert(0, "pos", range(1, len(t) + 1))
    return _records(t)


def team_series(season_df: pd.DataFrame, team: str, window: int = 5) -> list[dict]:
    """Per-match series for one team with rolling `window`-game averages."""
    long = team_matches(season_df)
    t = long[long.team == team].copy()
    t["cum_pts"] = t.pts.cumsum()
    t["matchday"] = range(1, len(t) + 1)
    for c in ("gf", "ga", "xgf", "xga", "pts", "sotf", "sota"):
        t[f"{c}_roll"] = t[c].rolling(window, min_periods=1).mean()
    t["date"] = t.date.dt.strftime("%Y-%m-%d")
    return _records(t.drop(columns=["match_idx"]))


def h2h(df: pd.DataFrame, a: str, b: str, n: int = 10) -> list[dict]:
    m = df[((df.home == a) & (df.away == b)) | ((df.home == b) & (df.away == a))].tail(n)
    out = m[["date", "season", "home", "away", "hg", "ag"]].copy()
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    return _records(out.iloc[::-1])


# --------------------------------------------------------------------------- Elo
def elo_ratings(df: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0,
                carry: float = 0.8) -> tuple[pd.DataFrame, dict[str, float]]:
    """Walk through matches chronologically. Returns per-match pre-match Elo
    (columns elo_h, elo_a) and the final ratings."""
    ratings: dict[str, float] = {}
    pre_h = np.empty(len(df))
    pre_a = np.empty(len(df))
    season = None
    for i, row in enumerate(df.itertuples(index=False)):
        if row.season != season:
            season = row.season
            if ratings:  # regress to the mean between seasons
                mean = np.mean(list(ratings.values()))
                ratings = {t: mean + carry * (r - mean) for t, r in ratings.items()}
        default = (min(ratings.values()) if ratings else 1500.0)  # promoted sides start low
        rh = ratings.setdefault(row.home, default)
        ra = ratings.setdefault(row.away, default)
        pre_h[i], pre_a[i] = rh, ra
        exp_h = 1 / (1 + 10 ** ((ra - rh - home_adv) / 400))
        score = 1.0 if row.hg > row.ag else 0.5 if row.hg == row.ag else 0.0
        gd = abs(row.hg - row.ag)
        mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = k * mult * (score - exp_h)
        ratings[row.home] = rh + delta
        ratings[row.away] = ra - delta
    return pd.DataFrame({"elo_h": pre_h, "elo_a": pre_a}, index=df.index), ratings


# --------------------------------------------------------------------------- ML features
FORM_STATS = ("gf", "ga", "sotf", "sota", "pts")
XG_STATS = ("xgf", "xga")          # only published for recent seasons -> may be NaN
MAX_REST = 14.0                    # days; longer breaks are capped
DEFAULT_REST = 7.0


def _filled(t: pd.DataFrame, c: str) -> pd.Series:
    """Shots on target, with a goals-based stand-in for the rare rows that lack them."""
    if c in ("sotf", "sota"):
        return t[c].fillna(t["gf" if c == "sotf" else "ga"] * 3)
    return t[c]


def prematch_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """For every match, each side's average over its previous `window` games
    (any venue, across seasons), days of rest, and pre-match Elo. Only uses
    information available before kick-off, so it is safe for training and backtesting."""
    long = team_matches(df)
    long = long.sort_values(["team", "date", "match_idx"])
    for c in FORM_STATS + XG_STATS:
        src = _filled(long, c)
        long[f"{c}_f"] = src.groupby(long.team).transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["n_prev"] = long.groupby("team").cumcount()
    rest = long.groupby("team").date.diff().dt.days
    long["rest"] = rest.clip(upper=MAX_REST).fillna(DEFAULT_REST)
    feats = [f"{c}_f" for c in FORM_STATS + XG_STATS] + ["n_prev", "rest"]
    h = long[long.venue == "H"].set_index("match_idx")[feats].add_prefix("h_")
    a = long[long.venue == "A"].set_index("match_idx")[feats].add_prefix("a_")
    elo, _ = elo_ratings(df)
    out = pd.concat([h, a], axis=1).reindex(df.index)
    out["elo_diff"] = (elo.elo_h - elo.elo_a) / 100.0
    return out


def latest_form(df: pd.DataFrame, team: str, window: int = 5, long: pd.DataFrame | None = None,
                on_date=None) -> dict:
    """Feature vector for a team's *next* match (average of its last `window` games).
    `on_date` is the kick-off date if known, used for days of rest."""
    long = team_matches(df) if long is None else long
    mine = long[long.team == team]
    t = mine.tail(window)
    out = {f"{c}_f": (float(_filled(t, c).mean()) if len(t) else np.nan) for c in FORM_STATS + XG_STATS}
    out["n_prev"] = int(len(mine))
    rest = DEFAULT_REST
    if on_date is not None and len(mine):
        days = (pd.Timestamp(on_date).tz_localize(None).normalize() - mine.date.iloc[-1]).days
        if days > 0:
            rest = min(MAX_REST, float(days))
    out["rest"] = rest
    return out


def _records(df: pd.DataFrame) -> list[dict]:
    return df.replace({np.nan: None}).to_dict(orient="records")
