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
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics, engine, models, sources

ROOT = Path(__file__).resolve().parent.parent
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
MODEL_KEYS = ["sheet", "form5", "dc", "ml", "blend"]


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


def upcoming(ctx, fx_csv: pd.DataFrame | None, fpl: dict | None) -> list[dict]:
    rows = []
    if fpl:
        names = sources.fpl_team_names(fpl["bootstrap"], set(ctx.teams))
        for f in fpl["fixtures"]:
            if f["finished"] or f["started"] or not f["kickoff_time"]:
                continue
            rows.append({"home": names[f["team_h"]], "away": names[f["team_a"]],
                         "kickoff": f["kickoff_time"], "round": f["event"]})
    odds = {}
    if fx_csv is not None and len(fx_csv):
        sub = fx_csv[fx_csv["div"] == ctx.div]
        odds = {(r.home, r.away): r for r in sub.itertuples()}
        if not rows:
            rows = [{"home": r.home, "away": r.away,
                     "kickoff": r.date.isoformat() if pd.notna(r.date) else None, "round": None}
                    for r in sub.itertuples()]
    rows = sorted(rows, key=lambda r: r["kickoff"] or "")[:40]
    out = []
    for r in rows:
        if r["home"] not in ctx.teams or r["away"] not in ctx.teams:
            continue
        p = ctx.predict(r["home"], r["away"])["blend"]
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
    return out


def scorer_shares(ctx, bootstrap: dict) -> dict:
    """Per team: each outfield player's share of team xG and fitness. The browser turns
    this into anytime-scorer odds: P = 1 - exp(-team_xG * share * fitness)."""
    names = sources.fpl_team_names(bootstrap, set(ctx.teams))
    out = {}
    for tid, team in names.items():
        players = [p for p in bootstrap["elements"] if p["team"] == tid and p["element_type"] != 1]
        team_xg = sum(float(p["expected_goals"]) for p in players) or 1.0
        rows = []
        for p in players:
            xg = float(p["expected_goals"])
            if xg <= 0 or p["minutes"] == 0:
                continue
            cop = p["chance_of_playing_next_round"]
            rows.append({"id": p["id"], "name": p["web_name"], "xg": xg, "goals": p["goals_scored"],
                         "share": xg / team_xg, "availability": (100 if cop is None else cop) / 100,
                         "news": p["news"]})
        out[team] = sorted(rows, key=lambda r: -r["share"] * r["availability"])[:10]
    return out


def export_league(div: str, season: str, out: Path, fx_csv, fpl) -> dict:
    t0 = time.time()
    ctx = engine.get_context(div, season)
    s = ctx.season_df
    teams = ctx.teams
    wk = s.groupby(s.date.dt.to_period("W").dt.start_time).agg(games=("hg", "size"), hg=("hg", "sum"), ag=("ag", "sum"))
    is_current = season == sources.current_season_code()

    # expected goals for every ordered pair, for every model
    pairs = {}
    for h in teams:
        for a in teams:
            if h == a:
                continue
            lams = ctx.lambdas(h, a)
            pairs[f"{h}|{a}"] = {k: (None if v is None else [v[0], v[1]]) for k, v in lams.items()}

    h2h = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            rows = analytics.h2h(ctx.history, a, b)
            if rows:
                h2h[f"{a}|{b}"] = rows

    series = {t: analytics.team_series(s, t, 5) for t in teams}
    dc = [r for r in ctx.dc.ratings() if r["team"] in teams]

    payload = {
        "div": div, "season": season, "season_label": sources.season_label(season),
        "league": sources.LEAGUES[div], "updated": ctx.data_ts, "teams": teams,
        "stats": league_stats(s),
        "table": analytics.league_table(s), "home_table": analytics.league_table(s, "H"),
        "away_table": analytics.league_table(s, "A"),
        "trend": [{"week": d.strftime("%Y-%m-%d"), "gpg": (r.hg + r.ag) / r.games, "games": int(r.games)}
                  for d, r in wk.iterrows()],
        "elo": sorted(({"team": t, "elo": ctx.elo[t]} for t in teams if t in ctx.elo), key=lambda r: -r["elo"]),
        "dc_ratings": dc, "dc_home_adv": float(np.exp(ctx.dc.home)), "rho": ctx.dc.rho,
        "series": series, "pairs": pairs, "h2h": h2h,
        "fixtures": upcoming(ctx, fx_csv, fpl) if is_current else [],
        "scorers": scorer_shares(ctx, fpl["bootstrap"]) if (is_current and fpl) else None,
    }
    try:
        payload["backtest"] = engine.backtest(ctx)
    except Exception:
        traceback.print_exc()
        payload["backtest"] = None
    write(out / "data" / f"{div}_{season}.json", payload)
    print(f"  {div} {season}: {len(s)} matches, {len(teams)} teams in {time.time() - t0:.1f}s")
    return {"code": div, "season": season, "matches": len(s)}


# --------------------------------------------------------------------------- players
def export_players(out: Path, bootstrap: dict, with_history: bool = True) -> None:
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

    hist = {}
    if with_history:
        # only players with real attacking involvement or regular minutes, to keep it quick
        ids = [r["id"] for r in rows if r["minutes"] >= 180]
        for n, pid in enumerate(ids):
            try:
                h = sources.fpl_player_history(pid, force=True)
            except Exception:
                continue
            hist[pid] = [[g["round"], g["minutes"], g["goals_scored"], g["assists"],
                          float(g["expected_goals"]), float(g["expected_assists"]), g["total_points"],
                          teams.get(g["opponent_team"], "?"), g["was_home"]]
                         for g in h.get("history", [])[-10:]]
            if n % 50 == 0:
                time.sleep(0.5)  # be polite to the FPL API
    write(out / "data" / "player_history.json", {
        "fields": ["round", "minutes", "goals", "assists", "xg", "xa", "points", "opponent", "home"],
        "players": hist})
    print(f"  players: {len(rows)} ({len(hist)} with history)")


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
    for f in (ROOT / "frontend").iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
    (out / ".nojekyll").write_text("")

    cur = sources.current_season_code()
    seasons = [cur] + sources.previous_seasons(cur, a.seasons - 1)

    try:
        fx_csv = sources.load_fixtures_csv(force=True)
    except Exception as e:
        print("fixtures.csv unavailable:", e)
        fx_csv = None
    try:
        fpl = {"bootstrap": sources.fpl_bootstrap(force=True), "fixtures": sources.fpl_fixtures(force=True)}
    except Exception as e:
        print("FPL unavailable:", e)
        fpl = None

    index = {"generated": time.time(), "current_season": cur,
             "seasons": [{"code": s, "label": sources.season_label(s)} for s in seasons],
             "models": models.MODEL_NAMES, "leagues": [], "available": []}
    for div in a.leagues:
        index["leagues"].append({"code": div, **sources.LEAGUES[div]})
        for season in seasons:
            try:
                sources.load_season(div, season, force=(season == cur))
                info = export_league(div, season, out, fx_csv, fpl if div == "E0" else None)
                index["available"].append(f"{div}_{season}")
            except Exception:
                print(f"  {div} {season}: FAILED")
                traceback.print_exc()
    if fpl:
        export_players(out, fpl["bootstrap"], with_history=not a.no_player_history)
        index["players"] = True
    write(out / "data" / "index.json", index)
    print("done ->", out)


if __name__ == "__main__":
    main()
