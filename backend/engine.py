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
_backtests: dict[tuple, dict] = {}

# models that go into the blend (the sheet model is left out: it cannot predict
# early-season games and is badly over-confident on small samples)
BLEND_MODELS = ("form5", "dc", "ml", "gbm")
WEIGHT_SHARPNESS = 25.0     # how strongly the blend favours the better models
WEIGHT_PRIOR_N = 150        # matches needed before past accuracy is trusted much


class LeagueContext:
    def __init__(self, div: str, season: str, force: bool = False):
        self.div, self.season = div, season
        hist = sources.load_history(div, season, n_prev=3, force=force).reset_index(drop=True)
        self.history = hist
        self.season_df = hist[hist.season == season].reset_index(drop=True)
        self.loaded_at = time.time()
        self.data_ts = sources.cache_timestamp(f"{div}_{season}.csv") or self.loaded_at
        self.teams = sorted(set(self.season_df.home) | set(self.season_df.away))
        self.long = analytics.team_matches(hist)
        self.feats = analytics.prematch_features(hist)
        _, self.elo = analytics.elo_ratings(hist)
        self.dc = models.DixonColes().fit(hist)
        self.ml = models.FormML("glm").fit(hist, self.feats)
        self.gbm = models.FormML("gbm").fit(hist, self.feats)
        self.weights: dict[str, float] | None = None   # set by backtest(); None = equal

    # ------------------------------------------------------------------ predict
    def lambdas(self, home: str, away: str, on_date=None) -> dict[str, tuple[float, float] | None]:
        out = {
            "sheet": models.sheet_lambdas(self.season_df, home, away),
            "form5": models.form_lambdas(self.history, self.season_df, home, away, long=self.long),
            "dc": self.dc.lambdas(home, away),
            "ml": self.ml.lambdas_for(self.history, home, away, self.elo, long=self.long, on_date=on_date),
            "gbm": self.gbm.lambdas_for(self.history, home, away, self.elo, long=self.long, on_date=on_date),
        }
        out["blend"] = blend(out, self.weights)
        return out

    def predict(self, home: str, away: str, on_date=None) -> dict:
        res = {}
        for name, lam in self.lambdas(home, away, on_date).items():
            rho = self.dc.rho if name in ("dc", "blend") else 0.0
            res[name] = None if lam is None else models.markets(*lam, rho=rho)
        return res


def blend(lams: dict, weights: dict[str, float] | None = None) -> tuple[float, float] | None:
    """Weighted geometric mean of the available models' expected goals."""
    avail = [k for k in BLEND_MODELS if lams.get(k) is not None]
    if not avail:
        fallback = [v for v in lams.values() if v is not None]
        return fallback[0] if fallback else None
    w = np.array([(weights or {}).get(k, 1.0) for k in avail], float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(avail), 1 / len(avail))
    arr = np.log(np.array([lams[k] for k in avail], float))
    return float(np.exp(w @ arr[:, 0])), float(np.exp(w @ arr[:, 1]))


def accuracy_weights(ll_sum: dict[str, float], n: dict[str, int]) -> dict[str, float]:
    """Blend weights from each model's average log-loss on matches already played.
    With few matches the weights stay close to equal; they sharpen as evidence builds."""
    have = [k for k in BLEND_MODELS if n.get(k, 0) > 0]
    if not have:
        return {k: 1 / len(BLEND_MODELS) for k in BLEND_MODELS}
    mean = {k: ll_sum[k] / n[k] for k in have}
    best = min(mean.values())
    n_min = min(n[k] for k in have)
    t = WEIGHT_SHARPNESS * n_min / (n_min + WEIGHT_PRIOR_N)
    raw = {k: float(np.exp(-t * (mean[k] - best))) if k in mean else 0.0 for k in BLEND_MODELS}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()}


def get_context(div: str, season: str | None = None, force: bool = False) -> LeagueContext:
    season = season or sources.active_season(div)
    key = (div, season)
    with _lock:
        ctx = _contexts.get(key)
        stale = ctx and not sources.season_is_final(div, season) and \
            time.time() - ctx.loaded_at > sources.STALE_AFTER
        if ctx is None or force or stale:
            ctx = LeagueContext(div, season, force=force)
            _contexts[key] = ctx
        return ctx


# --------------------------------------------------------------------------- backtest
MODEL_ORDER = ["sheet", "form5", "dc", "ml", "gbm", "blend", "market"]


def walk_forward(hist: pd.DataFrame, feats: pd.DataFrame, eval_seasons: list[str],
                 xi: float = models.DC_XI) -> tuple[dict, dict]:
    """For each week of the evaluation seasons, refit every model on everything
    before that week and predict its matches. Nothing from the future leaks in:
    even the blend's weights only use matches that had already been played.

    Returns (predictions per model, final blend weights)."""
    ev = hist[hist.season.isin(eval_seasons)]
    weeks = ev.date.dt.to_period("W")
    preds = {n: [] for n in MODEL_ORDER}
    ll_sum = {k: 0.0 for k in BLEND_MODELS}
    ll_n = {k: 0 for k in BLEND_MODELS}

    for wk in sorted(weeks.unique()):
        idx = ev.index[weeks == wk]
        start = ev.loc[idx, "date"].min()
        train = hist[hist.date < start]
        if len(train) < 200:
            continue
        season_train = train[train.season == ev.loc[idx[0], "season"]]
        long = analytics.team_matches(train)
        dc = models.DixonColes(xi=xi).fit(train, ref_date=start)
        ml = models.FormML("glm").fit(train, feats.loc[train.index])
        gbm = models.FormML("gbm").fit(train, feats.loc[train.index])
        ml_l, gbm_l = ml.predict(feats.loc[idx]), gbm.predict(feats.loc[idx])
        weights = accuracy_weights(ll_sum, ll_n)       # from matches before this week only
        week_ll = []
        for j, i in enumerate(idx):
            r = hist.loc[i]
            y = 0 if r.hg > r.ag else 1 if r.hg == r.ag else 2
            enough = feats.loc[i, "h_n_prev"] >= 3 and feats.loc[i, "a_n_prev"] >= 3
            lams = {
                "sheet": models.sheet_lambdas(season_train, r.home, r.away),
                "form5": models.form_lambdas(train, season_train, r.home, r.away, long=long),
                "dc": dc.lambdas(r.home, r.away),
                "ml": tuple(ml_l[j]) if enough else None,
                "gbm": tuple(gbm_l[j]) if enough else None,
            }
            lams["blend"] = blend(lams, weights)
            over = r.hg + r.ag > 2.5
            for n, lam in lams.items():
                if lam is None:
                    continue
                mk = models.markets(*lam, rho=dc.rho if n in ("dc", "blend") else 0.0)
                p = [mk["home"]["p"], mk["draw"]["p"], mk["away"]["p"]]
                preds[n].append((i, p, y, mk["over"]["2.5"]["p"], over))
                if n in ll_sum:
                    week_ll.append((n, -np.log(max(p[y], 1e-9))))
            imp = models.implied_probs(r.odds_h, r.odds_d, r.odds_a)
            if imp:
                preds["market"].append((i, imp, y, None, over))
        for n, v in week_ll:                           # only now may this week inform the weights
            ll_sum[n] += v
            ll_n[n] += 1
    return preds, accuracy_weights(ll_sum, ll_n)


def score(preds: dict) -> list[dict]:
    summary = []
    for n in MODEL_ORDER:
        rows = preds.get(n) or []
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
    return summary


def backtest(ctx: LeagueContext) -> dict:
    """Walk-forward test over this season and the one before it. Also sets the
    context's blend weights from the result."""
    key = (ctx.div, ctx.season, ctx.data_ts)
    if key not in _backtests:
        hist = ctx.history
        seasons = [s for s in (sources.previous_seasons(ctx.season, 1)[0], ctx.season)
                   if (hist.season == s).any()]
        preds, weights = walk_forward(hist, ctx.feats, seasons)

        calib = {}
        bins = np.linspace(0, 1, 11)
        for n in ("blend", "dc", "ml", "market"):
            rows = preds[n]
            if not rows:
                continue
            ph = np.array([p[0] for _, p, _, _, _ in rows])
            yh = np.array([o == 0 for _, _, o, _, _ in rows], float)
            b = np.digitize(ph, bins) - 1
            calib[n] = [{"pred": float(ph[b == k].mean()), "obs": float(yh[b == k].mean()), "n": int((b == k).sum())}
                        for k in range(10) if (b == k).sum() >= 8]

        _backtests[key] = {
            "seasons": seasons,
            "season_label": " + ".join(sources.season_label(s) for s in seasons),
            "summary": score(preds), "calibration": calib, "blend_weights": weights,
            "ml_importance": [{**r, "label": models.FEATURE_LABELS.get(r["feature"], r["feature"])}
                              for r in ctx.ml.importance()],
        }
    out = _backtests[key]
    ctx.weights = out["blend_weights"]
    return out
