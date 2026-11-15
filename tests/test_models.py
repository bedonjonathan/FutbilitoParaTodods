"""Ratings, features and the blend, on a made-up league with known team strengths."""
import numpy as np
import pytest

from backend import analytics, engine, models


def test_dixon_coles_ranks_the_strong_team_first(league):
    dc = models.DixonColes().fit(league)
    order = [r["team"] for r in dc.ratings()]
    assert order[0] in ("T0", "T1") and order[-1] in ("T8", "T9")
    lh, la = dc.lambdas("T0", "T9")
    assert lh > la
    assert dc.lambdas("T0", "Nobody") is None


def test_elo_ranks_the_strong_team_first(league):
    _, elo = analytics.elo_ratings(league)
    assert elo["T0"] > elo["T9"]


def test_features_only_use_earlier_matches(league):
    """Changing a result must not change that match's own pre-match features."""
    base = analytics.prematch_features(league)
    i = len(league) - 50
    changed = league.copy()
    changed.loc[i, ["hg", "ag", "hst", "ast"]] = [9, 9, 20, 20]
    after = analytics.prematch_features(changed)
    assert np.allclose(base.loc[i].to_numpy(float), after.loc[i].to_numpy(float), equal_nan=True)
    assert not np.allclose(base.loc[i + 30:].to_numpy(float), after.loc[i + 30:].to_numpy(float), equal_nan=True)


def test_rest_days_are_capped(league):
    f = analytics.prematch_features(league)
    assert f.h_rest.max() <= analytics.MAX_REST and f.h_rest.min() >= 0


@pytest.mark.parametrize("kind", ["glm", "gbm"])
def test_ml_models_predict_sensible_goals(league, kind):
    feats = analytics.prematch_features(league)
    m = models.FormML(kind).fit(league, feats)
    assert all(f in feats.columns for f in m.features)
    assert not any("xg" in f for f in m.features) or kind == "glm" or feats[m.features].notna().any().all()
    pred = m.predict(feats.tail(40))
    assert pred.shape == (40, 2) and (pred > 0).all() and (pred < 6).all()
    _, elo = analytics.elo_ratings(league)
    lh, la = m.lambdas_for(league, "T0", "T9", elo)
    assert lh > la


def test_sheet_model_matches_the_spreadsheet_formula(league):
    s = league[league.season == "2627"]
    lh, la = models.sheet_lambdas(s, "T0", "T9")
    hs = s.groupby("home").agg(m=("hg", "size"), gf=("hg", "sum"), ga=("ag", "sum"))
    aw = s.groupby("away").agg(m=("ag", "size"), gf=("ag", "sum"), ga=("hg", "sum"))
    avg_home = (hs.gf / hs.m).mean()
    expect = (hs.loc["T0", "gf"] / hs.loc["T0", "m"] / avg_home) * (aw.loc["T9", "ga"] / aw.loc["T9", "m"] / avg_home) * avg_home
    assert lh == pytest.approx(max(expect, 0.05))


def test_blend_weights():
    w = engine.accuracy_weights({"form5": 110.0, "dc": 100.0, "ml": 101.0, "gbm": 102.0},
                                {k: 100 for k in engine.BLEND_MODELS})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["dc"] > w["ml"] > w["gbm"] > w["form5"]
    equal = engine.accuracy_weights({k: 0.0 for k in engine.BLEND_MODELS}, {k: 0 for k in engine.BLEND_MODELS})
    assert all(v == pytest.approx(0.25) for v in equal.values())


def test_blend_handles_missing_models():
    lams = {"sheet": (9.0, 9.0), "form5": None, "dc": (2.0, 1.0), "ml": (1.0, 1.0), "gbm": None}
    lh, la = engine.blend(lams, {"dc": 0.75, "ml": 0.25, "form5": 0.0, "gbm": 0.0})
    assert lh == pytest.approx(2 ** 0.75) and la == pytest.approx(1.0)
    assert engine.blend({"sheet": None, "dc": None}) is None


def test_walk_forward_never_sees_the_future(league):
    feats = analytics.prematch_features(league)
    preds, weights = engine.walk_forward(league, feats, ["2627"])
    assert sum(weights.values()) == pytest.approx(1.0)
    n = (league.season == "2627").sum()
    assert 0 < len(preds["dc"]) <= n
    rows = engine.score(preds)
    by = {r["model"]: r for r in rows}
    assert by["dc"]["log_loss"] < 1.0986          # better than a coin-flip between three outcomes
    assert all(0 <= r["accuracy"] <= 1 for r in rows)
