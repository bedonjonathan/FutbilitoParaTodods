"""FastAPI server: JSON API under /api, the dashboard UI at /."""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import analytics, engine, models, sources

app = FastAPI(title="Futbolito Analytics")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def clean(o):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, (pd.Timestamp, dt.date)):
        return o.isoformat()
    return o


def ok(payload):
    return JSONResponse(clean(payload))


def ctx_for(div: str, season: str | None):
    if div not in sources.LEAGUES:
        raise HTTPException(404, f"Unknown league {div}")
    try:
        return engine.get_context(div, season)
    except Exception as e:  # network / parse problems
        raise HTTPException(502, f"Could not load data: {e}")


# --------------------------------------------------------------------------- meta
@app.get("/api/leagues")
def leagues():
    cur = sources.current_season_code()
    seasons = [cur] + sources.previous_seasons(cur, 3)
    return ok({
        "leagues": [{"code": k, **v} for k, v in sources.LEAGUES.items()],
        "seasons": [{"code": s, "label": sources.season_label(s)} for s in seasons],
        "current_season": cur,
        "models": models.MODEL_NAMES,
    })


@app.post("/api/refresh")
def refresh(div: str = "E0", season: str | None = None):
    try:
        ctx = engine.get_context(div, season, force=True)
        if sources.LEAGUES[div]["players"]:
            sources.fpl_bootstrap(force=True)
            sources.fpl_fixtures(force=True)
        sources.load_fixtures_csv(force=True)
    except Exception as e:
        raise HTTPException(502, f"Refresh failed: {e}")
    return ok({"updated": ctx.data_ts, "matches": len(ctx.season_df)})


# --------------------------------------------------------------------------- overview
@app.get("/api/overview")
def overview(div: str = "E0", season: str | None = None):
    ctx = ctx_for(div, season)
    s = ctx.season_df
    n = len(s)
    stats = {}
    if n:
        stats = {
            "matches": n,
            "goals_per_game": (s.hg + s.ag).mean(),
            "home_goals_pg": s.hg.mean(), "away_goals_pg": s.ag.mean(),
            "home_win": (s.hg > s.ag).mean(), "draw": (s.hg == s.ag).mean(), "away_win": (s.hg < s.ag).mean(),
            "over25": ((s.hg + s.ag) > 2.5).mean(), "btts": ((s.hg > 0) & (s.ag > 0)).mean(),
            "has_xg": bool(s.hxg.notna().any()),
        }
    # goals per week trend
    wk = s.groupby(s.date.dt.to_period("W").dt.start_time).agg(
        games=("hg", "size"), goals=("hg", "sum"), ag=("ag", "sum"))
    trend = [{"week": d.strftime("%Y-%m-%d"), "gpg": (r.goals + r.ag) / r.games, "games": int(r.games)}
             for d, r in wk.iterrows()]
    elo = sorted(({"team": t, "elo": ctx.elo[t]} for t in ctx.teams if t in ctx.elo),
                 key=lambda r: -r["elo"])
    dc = [r for r in ctx.dc.ratings() if r["team"] in ctx.teams]
    return ok({
        "league": sources.LEAGUES[div], "season": ctx.season, "season_label": sources.season_label(ctx.season),
        "updated": ctx.data_ts, "teams": ctx.teams, "stats": stats,
        "table": analytics.league_table(s), "home_table": analytics.league_table(s, "H"),
        "away_table": analytics.league_table(s, "A"),
        "trend": trend, "elo": elo, "dc_ratings": dc, "dc_home_adv": float(np.exp(ctx.dc.home)),
    })


@app.get("/api/team")
def team(team: str, div: str = "E0", season: str | None = None, window: int = Query(5, ge=1, le=20)):
    ctx = ctx_for(div, season)
    if team not in ctx.teams:
        raise HTTPException(404, f"{team} not in this season")
    series = analytics.team_series(ctx.season_df, team, window)
    dc = next((r for r in ctx.dc.ratings() if r["team"] == team), None)
    return ok({"team": team, "series": series, "dc": dc, "elo": ctx.elo.get(team),
               "form": analytics.latest_form(ctx.history, team, window)})


# --------------------------------------------------------------------------- predictions
def _scorers(ctx, home: str, away: str, lam_home: float, lam_away: float) -> dict | None:
    """Anytime-scorer chances (Premier League only, from FPL xG shares)."""
    if not sources.LEAGUES[ctx.div]["players"] or ctx.season != sources.current_season_code():
        return None
    try:
        bs = sources.fpl_bootstrap()
    except Exception:
        return None
    names = sources.fpl_team_names(bs, set(ctx.teams))
    out = {}
    for side, team, lam in (("home", home, lam_home), ("away", away, lam_away)):
        tid = next((i for i, n in names.items() if n == team), None)
        players = [p for p in bs["elements"] if p["team"] == tid and p["element_type"] != 1]
        team_xg = sum(float(p["expected_goals"]) for p in players) or 1.0
        games = max(1, int((ctx.season_df.home == team).sum() + (ctx.season_df.away == team).sum()))
        rows = []
        for p in players:
            xg = float(p["expected_goals"])
            if xg <= 0 or p["minutes"] == 0:
                continue
            avail = (p["chance_of_playing_next_round"] if p["chance_of_playing_next_round"] is not None else 100) / 100
            mins_share = min(1.0, p["minutes"] / (games * 90))
            share = xg / team_xg
            exp_goals = lam * share * avail
            rows.append({"id": p["id"], "name": p["web_name"], "xg": xg, "goals": p["goals_scored"],
                         "minutes_share": mins_share, "availability": avail,
                         "p_score": 1 - math.exp(-exp_goals), "news": p["news"]})
        out[side] = sorted(rows, key=lambda r: -r["p_score"])[:8]
    return out


def _market_odds(div: str, home: str, away: str, season_df: pd.DataFrame):
    try:
        fx = sources.load_fixtures_csv()
        row = fx[(fx["div"] == div) & (fx.home == home) & (fx.away == away)]
        if len(row):
            r = row.iloc[0]
            return {"source": "upcoming", "odds": [r.odds_h, r.odds_d, r.odds_a],
                    "implied": models.implied_probs(r.odds_h, r.odds_d, r.odds_a)}
    except Exception:
        pass
    return None


@app.get("/api/predict")
def predict(home: str, away: str, div: str = "E0", season: str | None = None):
    ctx = ctx_for(div, season)
    if home == away:
        raise HTTPException(400, "Pick two different teams")
    for t in (home, away):
        if t not in ctx.teams:
            raise HTTPException(404, f"{t} not in this season")
    preds = ctx.predict(home, away)
    blend = preds.get("blend")
    return ok({
        "home": home, "away": away, "models": preds, "model_names": models.MODEL_NAMES,
        "h2h": analytics.h2h(ctx.history, home, away),
        "form": {
            "home": analytics.team_series(ctx.history[ctx.history.season == ctx.season], home)[-5:],
            "away": analytics.team_series(ctx.history[ctx.history.season == ctx.season], away)[-5:],
        },
        "elo": {"home": ctx.elo.get(home), "away": ctx.elo.get(away)},
        "market": _market_odds(div, home, away, ctx.season_df),
        "scorers": _scorers(ctx, home, away, blend["lambda_home"], blend["lambda_away"]) if blend else None,
    })


@app.get("/api/fixtures")
def fixtures(div: str = "E0", limit: int = Query(20, ge=1, le=60)):
    ctx = ctx_for(div, None)
    rows = []
    if sources.LEAGUES[div]["players"]:
        try:
            bs = sources.fpl_bootstrap()
            names = sources.fpl_team_names(bs, set(ctx.teams))
            for f in sources.fpl_fixtures():
                if f["finished"] or f["started"] or not f["kickoff_time"]:
                    continue
                rows.append({"home": names[f["team_h"]], "away": names[f["team_a"]],
                             "kickoff": f["kickoff_time"], "round": f["event"]})
        except Exception:
            pass
    try:
        fx = sources.load_fixtures_csv()
        fx = fx[fx["div"] == div]
        odds = {(r.home, r.away): r for r in fx.itertuples()}
        if not rows:
            rows = [{"home": r.home, "away": r.away, "kickoff": r.date.isoformat() if pd.notna(r.date) else None,
                     "round": None} for r in fx.itertuples()]
    except Exception:
        odds = {}
    rows = sorted(rows, key=lambda r: r["kickoff"] or "")[:limit]
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
    return ok({"fixtures": out, "updated": ctx.data_ts})


@app.get("/api/backtest")
def backtest(div: str = "E0", season: str | None = None):
    ctx = ctx_for(div, season)
    return ok(engine.backtest(ctx))


# --------------------------------------------------------------------------- players (FPL)
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@app.get("/api/players")
def players():
    try:
        bs = sources.fpl_bootstrap()
    except Exception as e:
        raise HTTPException(502, f"FPL unavailable: {e}")
    teams = {t["id"]: t["name"] for t in bs["teams"]}
    rows = []
    for p in bs["elements"]:
        if p["minutes"] <= 0:
            continue
        n90 = p["minutes"] / 90
        xg, xa = float(p["expected_goals"]), float(p["expected_assists"])
        rows.append({
            "id": p["id"], "name": p["web_name"], "full_name": f'{p["first_name"]} {p["second_name"]}',
            "team": teams[p["team"]], "pos": POS.get(p["element_type"], "?"),
            "minutes": p["minutes"], "starts": p["starts"], "goals": p["goals_scored"], "assists": p["assists"],
            "xg": xg, "xa": xa, "xgi": xg + xa, "g_minus_xg": p["goals_scored"] - xg,
            "xg90": xg / n90 if n90 else 0, "xa90": xa / n90 if n90 else 0,
            "shots_proxy_threat": float(p["threat"]), "creativity": float(p["creativity"]),
            "form": float(p["form"]), "points": p["total_points"], "price": p["now_cost"] / 10,
            "selected": float(p["selected_by_percent"]), "status": p["status"], "news": p["news"],
            "clean_sheets": p["clean_sheets"], "saves": p["saves"], "bonus": p["bonus"],
            "defcon": p.get("defensive_contribution", 0),
        })
    return ok({"players": rows, "teams": sorted(teams.values())})


@app.get("/api/player/{pid}")
def player(pid: int):
    try:
        h = sources.fpl_player_history(pid)
        teams = {t["id"]: t["name"] for t in sources.fpl_bootstrap()["teams"]}
    except Exception as e:
        raise HTTPException(502, f"FPL unavailable: {e}")
    hist = [{
        "round": g["round"], "minutes": g["minutes"], "goals": g["goals_scored"], "assists": g["assists"],
        "xg": float(g["expected_goals"]), "xa": float(g["expected_assists"]), "points": g["total_points"],
        "opponent": teams.get(g["opponent_team"], "?"), "home": g["was_home"], "kickoff": g["kickoff_time"],
    } for g in h.get("history", [])]
    return ok({"history": hist, "upcoming": h.get("fixtures", [])[:5]})


# --------------------------------------------------------------------------- UI
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
