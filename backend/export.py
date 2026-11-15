"""Build the static site: download data, fit every model, write JSON next to the UI.

    python -m backend.export                 # all leagues -> site/
    python -m backend.export --leagues E0 SP1 --seasons 1

GitHub Actions runs this on a schedule and publishes `site/` to GitHub Pages, so the
website itself is plain files (no server needed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics, engine, models, sources

ROOT = Path(__file__).resolve().parent.parent
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 5)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return o.isoformat()
    return o


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), separators=(",", ":"), ensure_ascii=False))


# --------------------------------------------------------------------------- league
def league_stats(s: pd.DataFrame) -> dict:
    if not len(s):
        return {}
    return {
        "matches": len(s), "goals_per_game": (s.hg + s.ag).mean(),
        "home_goals_pg": s.hg.mean(), "away_goals_pg": s.ag.mean(),
        "home_win": (s.hg > s.ag).mean(), "draw": (s.hg == s.ag).mean(), "away_win": (s.hg < s.ag).mean(),
        "over25": ((s.hg + s.ag) > 2.5).mean(), "btts": ((s.hg > 0) & (s.ag > 0)).mean(),
        "has_xg": bool(s.hxg.notna().any()),
    }


WARNINGS: list[str] = []


def warn(msg: str) -> None:
    """Shown in the build log, as a GitHub annotation, and stored in index.json."""
    WARNINGS.append(msg)
    print(f"::warning::{msg}")


def upcoming(ctx, fx_csv: pd.DataFrame | None, fpl: dict | None) -> list[dict]:
    rows = []
    if fpl:
        names, unmatched = sources.fpl_team_names(fpl["bootstrap"], set(ctx.teams))
        if unmatched:
            warn(f"{ctx.div}: FPL teams with no match in the results data: {', '.join(unmatched)} "
                 f"- their fixtures and scorers are missing (add an alias in sources.FPL_TO_FDUK)")
        for f in fpl["fixtures"]:
            if f["finished"] or f["started"] or not f["kickoff_time"]:
                continue
            if f["team_h"] not in names or f["team_a"] not in names:
                continue
            rows.append({"home": names[f["team_h"]], "away": names[f["team_a"]],
                         "kickoff": f["kickoff_time"], "round": f["event"]})
    odds = {}
    if fx_csv is not None and len(fx_csv):
        sub = fx_csv[fx_csv["div"] == ctx.div]
        odds = {(r.home, r.away): r for r in sub.itertuples()}
        if not rows:
            rows = [{"home": r.home, "away": r.away,
                     "kickoff": r.kickoff if isinstance(r.kickoff, str) else None, "round": None}
                    for r in sub.itertuples()]
    rows = sorted(rows, key=lambda r: r["kickoff"] or "")[:40]
    out, dropped = [], []
    for r in rows:
        if r["home"] not in ctx.teams or r["away"] not in ctx.teams:
            dropped.append(f"{r['home']} v {r['away']}")
            continue
        p = ctx.predict(r["home"], r["away"], on_date=r["kickoff"])["blend"]
        if p is None:
            continue
        item = {**r, "lambda_home": p["lambda_home"], "lambda_away": p["lambda_away"],
                "p": [p["home"]["p"], p["draw"]["p"], p["away"]["p"]],
                "over25": p["over"]["2.5"]["p"], "btts": p["btts_yes"]["p"],
                "top_score": p["top_scores"][0]["score"]}
        o = odds.get((r["home"], r["away"]))
        if o is not None:
            imp = models.implied_probs(o.odds_h, o.odds_d, o.odds_a)
            if imp:
                item["odds"] = [o.odds_h, o.odds_d, o.odds_a]
                item["market_p"] = imp
                item["edge"] = [a - b for a, b in zip(item["p"], imp)]
        out.append(item)
    if dropped:
        warn(f"{ctx.div}: {len(dropped)} fixtures skipped, team not in this season's results: {'; '.join(dropped[:5])}")
    return out


def scorer_shares(ctx, fpl: dict, recent_minutes: dict[int, float]) -> dict:
    """Per team: each outfield player's expected share of the team's goals.

    share = (player xG per 90) x (expected minutes / 90) / (team xG per match)

    Expected minutes come from the player's last 5 gameweeks, so a striker who has
    lost his place (or a substitute who has just won one) is priced on current
    playing time rather than the season total. The browser then computes
    P(scores) = 1 - exp(-team expected goals x share x fitness)."""
    bootstrap = fpl["bootstrap"]
    names, _ = sources.fpl_team_names(bootstrap, set(ctx.teams))
    played = {}
    for f in fpl["fixtures"]:
        if f["finished"]:
            for k in ("team_h", "team_a"):
                played[f[k]] = played.get(f[k], 0) + 1
    out = {}
    for tid, team in names.items():
        games = max(1, played.get(tid, 0))
        players = [p for p in bootstrap["elements"] if p["team"] == tid and p["element_type"] != 1]
        team_xg_pm = sum(float(p["expected_goals"]) for p in players) / games
        if team_xg_pm <= 0:
            continue
        rows = []
        for p in players:
            xg = float(p["expected_goals"])
            if xg <= 0 or p["minutes"] < 45:
                continue
            exp_min = recent_minutes.get(p["id"], p["minutes"] / games)
            cop = p["chance_of_playing_next_round"]
            rows.append({"id": p["id"], "name": p["web_name"], "xg": xg, "goals": p["goals_scored"],
                         "share": (xg / p["minutes"] * 90) * (min(exp_min, 90) / 90) / team_xg_pm,
                         "exp_minutes": exp_min, "availability": (100 if cop is None else cop) / 100,
                         "pens": p.get("penalties_order") == 1, "news": p["news"]})
        total = sum(r["share"] for r in rows)
        if total > 1:      # expected minutes can't add up to more than the team's goals
            for r in rows:
                r["share"] /= total
        out[team] = sorted(rows, key=lambda r: -r["share"] * r["availability"])[:10]
    return out


def export_league(div: str, season: str, out: Path, fx_csv, fpl, recent_minutes, is_active: bool) -> None:
    t0 = time.time()
    ctx = engine.get_context(div, season)
    s = ctx.season_df
    if not len(s):
        raise ValueError("no played matches")
    teams = ctx.teams
    wk = s.groupby(s.date.dt.to_period("W").dt.start_time).agg(games=("hg", "size"), hg=("hg", "sum"), ag=("ag", "sum"))

    try:   # run first: it also sets the blend weights used for every prediction below
        bt = engine.backtest(ctx)
    except Exception:
        traceback.print_exc()
        warn(f"{div} {season}: backtest failed, blend uses equal weights")
        bt = None

    # expected goals for every ordered pair, for every model
    pairs = {}
    for h in teams:
        for a in teams:
            if h != a:
                pairs[f"{h}|{a}"] = {k: (None if v is None else [v[0], v[1]]) for k, v in ctx.lambdas(h, a).items()}

    h2h = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            rows = analytics.h2h(ctx.history, a, b)
            if rows:
                h2h[f"{a}|{b}"] = rows

    stale = f"{div}_{season}.csv" in sources.STALE_SERVED
    if stale:
        warn(f"{div} {season}: download failed, using results saved from an earlier run")
    payload = {
        "div": div, "season": season, "season_label": sources.season_label(season),
        "league": sources.LEAGUES[div], "updated": ctx.data_ts, "teams": teams,
        "data_through": s.date.max().strftime("%Y-%m-%d"), "stale": stale,
        "stats": league_stats(s),
        "table": analytics.league_table(s), "home_table": analytics.league_table(s, "H"),
        "away_table": analytics.league_table(s, "A"),
        "trend": [{"week": d.strftime("%Y-%m-%d"), "gpg": (r.hg + r.ag) / r.games, "games": int(r.games)}
                  for d, r in wk.iterrows()],
        "elo": sorted(({"team": t, "elo": ctx.elo[t]} for t in teams if t in ctx.elo), key=lambda r: -r["elo"]),
        "dc_ratings": [r for r in ctx.dc.ratings() if r["team"] in teams],
        "dc_home_adv": float(np.exp(ctx.dc.home)), "rho": ctx.dc.rho,
        "series": {t: analytics.team_series(s, t, 5) for t in teams}, "pairs": pairs, "h2h": h2h,
        "fixtures": upcoming(ctx, fx_csv, fpl) if is_active else [],
        "scorers": scorer_shares(ctx, fpl, recent_minutes) if (is_active and fpl) else None,
        "backtest": bt,
    }
    write(out / "data" / f"{div}_{season}.json", payload)
    print(f"  {div} {season}: {len(s)} matches, {len(teams)} teams in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------- players
def export_players(out: Path, bootstrap: dict, with_history: bool = True) -> tuple[int, dict[int, float]]:
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    rows = []
    for p in bootstrap["elements"]:
        if p["minutes"] <= 0:
            continue
        n90 = p["minutes"] / 90
        xg, xa = float(p["expected_goals"]), float(p["expected_assists"])
        rows.append({
            "id": p["id"], "name": p["web_name"], "full_name": f'{p["first_name"]} {p["second_name"]}',
            "team": teams[p["team"]], "pos": POS.get(p["element_type"], "?"),
            "minutes": p["minutes"], "starts": p["starts"], "goals": p["goals_scored"], "assists": p["assists"],
            "xg": xg, "xa": xa, "xgi": xg + xa, "g_minus_xg": p["goals_scored"] - xg,
            "xg90": xg / n90, "xa90": xa / n90, "form": float(p["form"]), "points": p["total_points"],
            "price": p["now_cost"] / 10, "selected": float(p["selected_by_percent"]),
            "status": p["status"], "news": p["news"],
        })
    write(out / "data" / "players.json", {"players": rows, "teams": sorted(teams.values())})

    hist, recent = {}, {}
    if with_history:
        # only players with regular minutes, to keep the number of requests modest
        ids = [r["id"] for r in rows if r["minutes"] >= 180]
        failed = 0
        for pid in ids:
            try:
                h = sources.fpl_player_history(pid, force=True)
            except Exception:
                failed += 1
                continue
            games = h.get("history", [])
            hist[pid] = [[g["round"], g["minutes"], g["goals_scored"], g["assists"],
                          float(g["expected_goals"]), float(g["expected_assists"]), g["total_points"],
                          teams.get(g["opponent_team"], "?"), g["was_home"]]
                         for g in games[-10:]]
            if games:
                last = games[-5:]
                recent[pid] = sum(g["minutes"] for g in last) / len(last)
            time.sleep(0.15)  # be polite to the FPL API
        if failed:
            warn(f"players: game-by-game history failed for {failed} of {len(ids)} players")
    write(out / "data" / "player_history.json", {
        "fields": ["round", "minutes", "goals", "assists", "xg", "xa", "points", "opponent", "home"],
        "players": hist})
    print(f"  players: {len(rows)} ({len(hist)} with history)")
    return len(rows), recent


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--leagues", nargs="*", default=list(sources.LEAGUES))
    ap.add_argument("--seasons", type=int, default=2, help="current season + N-1 previous")
    ap.add_argument("--no-player-history", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)

    # copy the UI
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "frontend", out, dirs_exist_ok=True)
    (out / ".nojekyll").write_text("")

    try:
        fx_csv = sources.load_fixtures_csv(force=True)
    except Exception as e:
        warn(f"upcoming-fixtures feed unavailable ({e}); odds and non-PL fixtures are missing")
        fx_csv = None
    try:
        fpl = {"bootstrap": sources.fpl_bootstrap(force=True), "fixtures": sources.fpl_fixtures(force=True)}
        if {"fpl_bootstrap.json", "fpl_fixtures.json"} & sources.STALE_SERVED:
            warn("FPL download failed, using player data saved from an earlier run")
    except Exception as e:
        warn(f"FPL unavailable ({e}); players, PL fixtures and scorer odds are missing")
        fpl = None

    n_players, recent_minutes = 0, {}
    if fpl:
        n_players, recent_minutes = export_players(out, fpl["bootstrap"], with_history=not a.no_player_history)

    index = {"generated": time.time(), "models": models.MODEL_NAMES, "leagues": [], "available": [],
             "default_season": {}, "kept_from_previous_build": [], "players": n_players > 0}
    all_seasons: dict[str, None] = {}
    for div in a.leagues:
        index["leagues"].append({"code": div, **sources.LEAGUES[div]})
        try:
            sources.load_season(div, sources.current_season_code(), force=True)
        except Exception:
            pass                      # e.g. July: the new season's file does not exist yet
        cal = sources.current_season_code()
        # A page already published for the calendar season proves that season has started,
        # so a failed download today must not be mistaken for "season not begun yet".
        active = cal if (out / "data" / f"{div}_{cal}.json").exists() else sources.active_season(div)
        seasons = [active] + sources.previous_seasons(active, a.seasons - 1)
        for season in seasons:
            key = f"{div}_{season}"
            try:
                export_league(div, season, out, fx_csv, fpl if div == "E0" else None,
                              recent_minutes, is_active=(season == active))
            except Exception as e:
                traceback.print_exc()
                if (out / "data" / f"{key}.json").exists():   # keep what the last build published
                    warn(f"{key}: build failed ({e}); keeping the data from the previous build")
                    index["kept_from_previous_build"].append(key)
                else:
                    warn(f"{key}: build failed ({e}) and there is no earlier copy; league/season omitted")
                    continue
            index["available"].append(key)
            all_seasons[season] = None
            index["default_season"].setdefault(div, season)

    index["seasons"] = [{"code": s, "label": sources.season_label(s)} for s in sorted(all_seasons, reverse=True)]
    index["current_season"] = index["default_season"].get("E0") or (index["seasons"][0]["code"] if index["seasons"] else None)
    index["warnings"] = WARNINGS
    write(out / "data" / "index.json", index)

    # never publish a site with no Premier League data: failing here keeps the last good site live
    required = a.leagues[0]
    if required not in index["default_season"]:
        print(f"::error::No data could be built for {required}; not publishing.")
        sys.exit(1)
    print(f"done -> {out}  ({len(index['available'])} league-seasons, {len(WARNINGS)} warnings)")


if __name__ == "__main__":
    main()
