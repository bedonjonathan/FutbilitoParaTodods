"""Goal-expectation models. Every model returns (lambda_home, lambda_away);
`markets()` turns those into a scoreline matrix and betting-style markets,
the same pipeline as the original spreadsheet (rows 38-48 of the Goals sheet).

Models
------
sheet   Exact port of the spreadsheet: season home/away attack & defence strength.
form5   Same structure, but strengths come from each team's LAST 5 matches, and
        goals are blended 50/50 with xG when xG is available.
dc      Dixon-Coles: maximum-likelihood attack/defence/home-advantage ratings over
        several seasons with exponential time-decay (recent games count more) and
        the low-score correlation correction (rho) for 0-0, 1-0, 0-1, 1-1.
ml      Machine learning: two Poisson GLMs (home goals, away goals) trained on
        every past match using last-5-game form (goals, shots on target, points)
        and Elo ratings as features.
blend   Average of the models' expected goals (an ensemble).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import analytics

MAX_GOALS = 10
MODEL_NAMES = {
    "sheet": "Your sheet (season home/away)",
    "form5": "Last-5 form (+xG)",
    "dc": "Dixon-Coles (time-weighted)",
    "ml": "ML Poisson (form + Elo)",
    "blend": "Blend (ensemble)",
}


# --------------------------------------------------------------------------- markets
def score_matrix(lh: float, la: float, rho: float = 0.0, n: int = MAX_GOALS) -> np.ndarray:
    """m[h, a] = P(home scores h, away scores a)."""
    g = np.arange(n + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    if rho:
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def american(p: float) -> int | None:
    """Fair American odds. (The sheet used (1/p - 1)*100, which is only right for
    underdogs; favourites need the negative form.)"""
    if p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def _price(p: float) -> dict:
    p = float(p)
    return {"p": p, "decimal": (1 / p) if p > 0 else None, "american": american(p)}


def markets(lh: float, la: float, rho: float = 0.0) -> dict:
    lh, la, rho = float(lh), float(la), float(rho)
    m = score_matrix(lh, la, rho)
    h_idx, a_idx = np.indices(m.shape)
    total = h_idx + a_idx
    out = {
        "lambda_home": lh, "lambda_away": la, "rho": rho,
        "home": _price(m[h_idx > a_idx].sum()),
        "draw": _price(m[h_idx == a_idx].sum()),
        "away": _price(m[h_idx < a_idx].sum()),
        "btts_yes": _price(m[(h_idx > 0) & (a_idx > 0)].sum()),
        "home_cs": _price(m[:, 0].sum()),
        "away_cs": _price(m[0, :].sum()),
        "over": {str(t): _price(m[total > t].sum()) for t in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "under": {str(t): _price(m[total < t].sum()) for t in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "home_goals_dist": m.sum(axis=1)[:7].tolist(),
        "away_goals_dist": m.sum(axis=0)[:7].tolist(),
        "matrix": m[:7, :7].tolist(),
    }
    flat = [(m[h, a], f"{h}-{a}") for h in range(7) for a in range(7)]
    out["top_scores"] = [{"score": s, **_price(p)} for p, s in sorted(flat, reverse=True)[:8]]
    return out


# --------------------------------------------------------------------------- sheet model
def _strength_inputs(season_df: pd.DataFrame):
    hs = season_df.groupby("home").agg(m=("hg", "size"), gf=("hg", "sum"), ga=("ag", "sum"))
    as_ = season_df.groupby("away").agg(m=("ag", "size"), gf=("ag", "sum"), ga=("hg", "sum"))
    return hs, as_


def sheet_lambdas(season_df: pd.DataFrame, home: str, away: str, min_games: int = 2):
    """Port of B31:B40 in the sheet. The league averages are averages of the per-team
    per-game rates (as the sheet's AVERAGE(W4:W24) does)."""
    hs, as_ = _strength_inputs(season_df)
    if home not in hs.index or away not in as_.index:
        return None
    if hs.loc[home, "m"] < min_games or as_.loc[away, "m"] < min_games:
        return None
    avg_home_gpg = (hs.gf / hs.m).mean()        # B31
    avg_away_gpg = (as_.gf / as_.m).mean()      # B32
    home_attack = (hs.loc[home, "gf"] / hs.loc[home, "m"]) / avg_home_gpg   # E31
    away_defence = (as_.loc[away, "ga"] / as_.loc[away, "m"]) / avg_home_gpg  # E32
    away_attack = (as_.loc[away, "gf"] / as_.loc[away, "m"]) / avg_away_gpg  # E33
    home_defence = (hs.loc[home, "ga"] / hs.loc[home, "m"]) / avg_away_gpg  # E34
    lh = home_attack * away_defence * avg_home_gpg   # B39
    la = away_attack * home_defence * avg_away_gpg   # B40
    return max(lh, 0.05), max(la, 0.05)


# --------------------------------------------------------------------------- form5
def form_lambdas(history: pd.DataFrame, season_df: pd.DataFrame, home: str, away: str,
                 window: int = 5, xg_weight: float = 0.5):
    long = analytics.team_matches(history)
    th = long[long.team == home].tail(window)
    ta = long[long.team == away].tail(window)
    if len(th) < 3 or len(ta) < 3:
        return None

    def rate(t, g, xg):
        goals = t[g].mean()
        if t[xg].notna().all():
            return (1 - xg_weight) * goals + xg_weight * t[xg].mean()
        return goals

    ref = season_df if len(season_df) >= 30 else history.tail(380)
    avg_h, avg_a = ref.hg.mean(), ref.ag.mean()
    avg_team = (avg_h + avg_a) / 2
    att_h, def_h = rate(th, "gf", "xgf") / avg_team, rate(th, "ga", "xga") / avg_team
    att_a, def_a = rate(ta, "gf", "xgf") / avg_team, rate(ta, "ga", "xga") / avg_team
    # shrink toward 1.0 - five games is a small sample
    k = len(th) / (len(th) + 3)
    att_h, def_h, att_a, def_a = (1 + k * (x - 1) for x in (att_h, def_h, att_a, def_a))
    return max(att_h * def_a * avg_h, 0.05), max(att_a * def_h * avg_a, 0.05)


# --------------------------------------------------------------------------- Dixon-Coles
class DixonColes:
    def __init__(self, xi: float = 0.0019, ridge: float = 0.05):
        self.xi = xi            # time decay per day (half-life ~ 1 year)
        self.ridge = ridge      # L2 penalty: keeps teams with few games near average

    def fit(self, df: pd.DataFrame, ref_date=None) -> "DixonColes":
        df = df.dropna(subset=["hg", "ag"])
        ref_date = ref_date or df.date.max()
        days = (pd.Timestamp(ref_date) - df.date).dt.days.clip(lower=0).to_numpy()
        w = np.exp(-self.xi * days)
        teams = sorted(set(df.home) | set(df.away))
        idx = {t: i for i, t in enumerate(teams)}
        hi = df.home.map(idx).to_numpy()
        ai = df.away.map(idx).to_numpy()
        hg = df.hg.to_numpy(float)
        ag = df.ag.to_numpy(float)
        n = len(teams)

        def nll(theta):
            att, dfn, home = theta[:n], theta[n:2 * n], theta[2 * n]
            log_lh = att[hi] + dfn[ai] + home
            log_la = att[ai] + dfn[hi]
            lh, la = np.exp(log_lh), np.exp(log_la)
            ll = w * (hg * log_lh - lh + ag * log_la - la)
            r_h = w * (hg - lh)
            r_a = w * (ag - la)
            g_att = np.bincount(hi, r_h, n) + np.bincount(ai, r_a, n)
            g_def = np.bincount(ai, r_h, n) + np.bincount(hi, r_a, n)
            g_home = r_h.sum()
            pen = self.ridge * (att @ att + dfn @ dfn)
            grad = -np.concatenate([g_att, g_def, [g_home]])
            grad[:2 * n] += 2 * self.ridge * theta[:2 * n]
            return -ll.sum() + pen, grad

        x0 = np.zeros(2 * n + 1)
        x0[2 * n] = 0.25
        res = minimize(nll, x0, jac=True, method="L-BFGS-B")
        att, dfn, home = res.x[:n], res.x[n:2 * n], res.x[2 * n]
        shift = att.mean()
        att, dfn = att - shift, dfn + shift   # identifiability: mean attack = 0
        self.teams, self.idx = teams, idx
        self.att, self.dfn, self.home = att, dfn, home

        # rho: 1-D fit on the low-score correction given the fitted rates
        lh = np.exp(att[hi] + dfn[ai] + home)
        la = np.exp(att[ai] + dfn[hi])

        def rho_nll(rho):
            tau = np.ones_like(lh)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1 - lh[m00] * la[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + la[m10] * rho
            tau[m11] = 1 - rho
            if (tau <= 0).any():
                return 1e9
            return -(w * np.log(tau)).sum()

        self.rho = float(minimize_scalar(rho_nll, bounds=(-0.2, 0.2), method="bounded").x)
        return self

    def lambdas(self, home: str, away: str):
        if home not in self.idx or away not in self.idx:
            return None
        h, a = self.idx[home], self.idx[away]
        return (float(np.exp(self.att[h] + self.dfn[a] + self.home)),
                float(np.exp(self.att[a] + self.dfn[h])))

    def ratings(self) -> list[dict]:
        rows = [{"team": t, "attack": float(np.exp(self.att[i])), "defence": float(np.exp(self.dfn[i]))}
                for t, i in self.idx.items()]
        return sorted(rows, key=lambda r: r["attack"] / r["defence"], reverse=True)


# --------------------------------------------------------------------------- ML
ML_FEATURES = ["elo_diff"] + [f"{s}_{c}_f" for s in ("h", "a") for c in analytics.FORM_STATS]


class FormML:
    """Poisson GLMs on last-5 form + Elo. Learns how much each recent-form signal
    actually matters, instead of hand-picking weights."""

    def fit(self, df: pd.DataFrame, feats: pd.DataFrame) -> "FormML":
        mask = (feats.h_n_prev >= 3) & (feats.a_n_prev >= 3)
        X = feats.loc[mask, ML_FEATURES].fillna(feats[ML_FEATURES].mean())
        self.fill = X.mean()
        self.home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.home_model.fit(X, df.loc[X.index, "hg"])
        self.away_model.fit(X, df.loc[X.index, "ag"])
        self.n_train = int(len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[ML_FEATURES].fillna(self.fill)
        return np.column_stack([self.home_model.predict(X), self.away_model.predict(X)])

    def lambdas_for(self, history: pd.DataFrame, home: str, away: str, elo: dict):
        if home not in elo or away not in elo:
            return None
        fh = analytics.latest_form(history, home)
        fa = analytics.latest_form(history, away)
        row = {"elo_diff": (elo[home] - elo[away]) / 100.0}
        row |= {f"h_{k}": v for k, v in fh.items()}
        row |= {f"a_{k}": v for k, v in fa.items()}
        lh, la = self.predict(pd.DataFrame([row]))[0]
        return float(lh), float(la)

    def importance(self) -> list[dict]:
        out = []
        for side, mdl in (("home goals", self.home_model), ("away goals", self.away_model)):
            coefs = mdl[-1].coef_
            for f, c in zip(ML_FEATURES, coefs):
                out.append({"target": side, "feature": f, "coef": float(c)})
        return out


FEATURE_LABELS = {
    "elo_diff": "Elo difference (home - away)",
    "h_gf_f": "Home: goals scored, last 5", "h_ga_f": "Home: goals conceded, last 5",
    "h_sotf_f": "Home: shots on target, last 5", "h_sota_f": "Home: shots on target against, last 5",
    "h_pts_f": "Home: points per game, last 5",
    "a_gf_f": "Away: goals scored, last 5", "a_ga_f": "Away: goals conceded, last 5",
    "a_sotf_f": "Away: shots on target, last 5", "a_sota_f": "Away: shots on target against, last 5",
    "a_pts_f": "Away: points per game, last 5",
}


def implied_probs(odds_h, odds_d, odds_a):
    """Bookmaker odds -> probabilities with the margin (overround) removed."""
    try:
        inv = np.array([1 / odds_h, 1 / odds_d, 1 / odds_a], dtype=float)
    except (TypeError, ZeroDivisionError):
        return None
    if not np.all(np.isfinite(inv)) or (inv <= 0).any():
        return None
    return (inv / inv.sum()).tolist()


def safe(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
