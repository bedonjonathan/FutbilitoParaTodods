"""Loads a league, fits every model once, and answers predictions / backtests.
Fitted contexts are memoised until the data is refreshed."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from . import analytics, models, sources

_lock = threading.Lock()
_contexts: dict[tuple[str, str], "LeagueContext"] = {}
_backtests: dict[tuple[str, str, float], dict] = {}


class LeagueContext:
    def __init__(self, div: str, season: str, force: bool = False):
        self.div, self.season = div, season
        self.history = sources.load_history(div, season, n_prev=3, force=force)
        self.season_df = self.history[self.history.season == season].reset_index(drop=True)
        self.loaded_at = time.time()
        self.data_ts = sources.cache_timestamp(f"{div}_{season}.csv") or self.loaded_at
        self.teams = sorted(set(self.season_df.home) | set(self.season_df.away))
        hist = self.history.reset_index(drop=True)
        self.history = hist
        self.long = analytics.team_matches(hist)
        self.feats = analytics.prematch_features(hist)
        _, self.elo = analytics.elo_ratings(hist)
        self.dc = models.DixonColes().fit(hist)
        self.ml = models.FormML().fit(hist, self.feats)

    # ------------------------------------------------------------------ predict
    def lambdas(self, home: str, away: str) -> dict[str, tuple[float, float] | None]:
        out = {
            "sheet": models.sheet_lambdas(self.season_df, home, away),
            "form5": models.form_lambdas(self.history, self.season_df, home, away, long=self.long),
            "dc": self.dc.lambdas(home, away),
            "ml": self.ml.lambdas_for(self.history, home, away, self.elo, long=self.long),
        }
        out["blend"] = _blend(out)
        return out

    def predict(self, home: str, away: str) -> dict:
        lams = self.lambdas(home, away)
        res = {}
        for name, lam in lams.items():
            if lam is None:
                res[name] = None
                continue
            rho = self.dc.rho if name in ("dc", "blend") else 0.0
            res[name] = models.markets(*lam, rho=rho)
        return res


def _blend(lams: dict) -> tuple[float, float] | None:
    vals = [v for k, v in lams.items() if v is not None and k != "sheet"] or \
           [v for v in lams.values() if v is not None]
    if not vals:
        return None
    arr = np.array(vals)
    return float(np.exp(np.log(arr[:, 0]).mean())), float(np.exp(np.log(arr[:, 1]).mean()))


def get_context(div: str, season: str | None = None, force: bool = False) -> LeagueContext:
    season = season or sources.current_season_code()
    key = (div, season)
    with _lock:
        ctx = _contexts.get(key)
        stale = ctx and season == sources.current_season_code() and \
            time.time() - ctx.loaded_at > sources.STALE_AFTER
        if ctx is None or force or stale:
            ctx = LeagueContext(div, season, force=force)
            _contexts[key] = ctx
        return ctx


# --------------------------------------------------------------------------- backtest
def backtest(ctx: LeagueContext) -> dict:
    """Walk-forward test: for each week of the season, refit on everything before
    that week and predict its matches. Nothing from the future leaks in."""
    key = (ctx.div, ctx.season, ctx.data_ts)
    if key in _backtests:
        return _backtests[key]

    hist, feats = ctx.history, ctx.feats
    season_idx = hist.index[hist.season == ctx.season]
    eval_season = ctx.season
    if len(season_idx) < 40:  # too early in the season: evaluate last season instead
        eval_season = sources.previous_seasons(ctx.season, 1)[0]
        season_idx = hist.index[hist.season == eval_season]
    ev = hist.loc[season_idx]
    weeks = ev.date.dt.to_period("W")

    names = ["sheet", "form5", "dc", "ml", "blend", "market"]
    preds = {n: [] for n in names}
    for wk in sorted(weeks.unique()):
        idx = ev.index[weeks == wk]
        start = ev.loc[idx, "date"].min()
        train = hist[hist.date < start]
        if len(train) < 200:
            continue
        season_train = train[train.season == eval_season]
        dc = models.DixonColes().fit(train, ref_date=start)
        ml = models.FormML().fit(train, feats.loc[train.index])
        ml_lams = ml.predict(feats.loc[idx])
        for j, i in enumerate(idx):
            r = hist.loc[i]
            outcome = 0 if r.hg > r.ag else 1 if r.hg == r.ag else 2
            lams = {
                "sheet": models.sheet_lambdas(season_train, r.home, r.away),
                "form5": models.form_lambdas(train, season_train, r.home, r.away),
                "dc": dc.lambdas(r.home, r.away),
                "ml": tuple(ml_lams[j]) if feats.loc[i, "h_n_prev"] >= 3 and feats.loc[i, "a_n_prev"] >= 3 else None,
            }
            lams["blend"] = _blend(lams)
            for n, lam in lams.items():
                if lam is None:
                    continue
                mk = models.markets(*lam, rho=dc.rho if n in ("dc", "blend") else 0.0)
                p = [mk["home"]["p"], mk["draw"]["p"], mk["away"]["p"]]
                preds[n].append((i, p, outcome, mk["over"]["2.5"]["p"], r.hg + r.ag > 2.5))
            imp = models.implied_probs(r.odds_h, r.odds_d, r.odds_a)
            if imp:
                preds["market"].append((i, imp, outcome, None, r.hg + r.ag > 2.5))

    summary = []
    for n in names:
        rows = preds[n]
        if not rows:
            continue
        P = np.array([p for _, p, _, _, _ in rows])
        y = np.array([o for _, _, o, _, _ in rows])
        onehot = np.eye(3)[y]
        pc = np.clip(P[np.arange(len(y)), y], 1e-9, 1)
        # ranked probability score (lower is better) - the standard for 1X2
        rps = (((np.cumsum(P, 1) - np.cumsum(onehot, 1))[:, :2]) ** 2).sum(1) / 2
        item = {
            "model": n, "label": models.MODEL_NAMES.get(n, "Bookmakers (avg odds)"),
            "n": int(len(y)),
            "accuracy": float((P.argmax(1) == y).mean()),
            "log_loss": float(-np.log(pc).mean()),
            "brier": float(((P - onehot) ** 2).sum(1).mean()),
            "rps": float(rps.mean()),
        }
        ou = [(p, hit) for _, _, _, p, hit in rows if p is not None]
        if ou:
            po = np.array([p for p, _ in ou])
            ho = np.array([h for _, h in ou], float)
            item["ou25_accuracy"] = float(((po > 0.5) == ho).mean())
        summary.append(item)

    # calibration of home-win probability for each model
    calib = {}
    bins = np.linspace(0, 1, 11)
    for n in ("blend", "dc", "ml", "market"):
        rows = preds[n]
        if not rows:
            continue
        ph = np.array([p[0] for _, p, _, _, _ in rows])
        yh = np.array([o == 0 for _, _, o, _, _ in rows], float)
        b = np.digitize(ph, bins) - 1
        pts = []
        for k in range(10):
            sel = b == k
            if sel.sum() >= 5:
                pts.append({"pred": float(ph[sel].mean()), "obs": float(yh[sel].mean()), "n": int(sel.sum())})
        calib[n] = pts

    out = {
        "season": eval_season, "season_label": sources.season_label(eval_season),
        "summary": summary, "calibration": calib,
        "ml_importance": [
            {**r, "label": models.FEATURE_LABELS.get(r["feature"], r["feature"])}
            for r in ctx.ml.importance()
        ],
    }
    _backtests[key] = out
    return out
