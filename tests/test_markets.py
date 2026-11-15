"""The odds maths, including the cells the original spreadsheet missed, and a check
that the browser (JavaScript) and the build (Python) produce the same numbers."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend import models

MATHS_JS = Path(__file__).resolve().parent.parent / "frontend" / "maths.js"
CASES = [(1.49, 0.86, -0.066), (2.6, 0.2, 0.0), (0.4, 0.5, -0.12), (3.2, 2.9, 0.05), (1.0, 1.0, 0.0)]


def test_probabilities_add_up():
    for lh, la, rho in CASES:
        m = models.markets(lh, la, rho)
        assert m["home"]["p"] + m["draw"]["p"] + m["away"]["p"] == pytest.approx(1.0)
        for line in ("0.5", "1.5", "2.5", "3.5", "4.5"):
            assert m["over"][line]["p"] + m["under"][line]["p"] == pytest.approx(1.0)


def test_home_win_includes_one_nil():
    # the sheet's B9 summed H40:N40 and so left out the 1-0 cell
    g = models.score_matrix(1.5, 1.0)
    h, a = np.indices(g.shape)
    assert models.markets(1.5, 1.0)["home"]["p"] == pytest.approx(g[h > a].sum())
    assert g[1, 0] > 0.05


def test_under_three_and_a_half_includes_nil_three():
    # the sheet's B18 left out the 0-3 cell
    g = models.score_matrix(1.5, 1.0)
    h, a = np.indices(g.shape)
    assert models.markets(1.5, 1.0)["under"]["3.5"]["p"] == pytest.approx(g[h + a <= 3].sum())


def test_american_odds():
    assert models.american(0.70) == -233      # favourite: negative
    assert models.american(0.25) == 300       # underdog: positive
    assert models.american(0.5) == -100
    assert models.american(0) is None and models.american(1) is None


def test_implied_probabilities_remove_the_margin():
    p = models.implied_probs(2.0, 3.4, 3.8)
    assert sum(p) == pytest.approx(1.0)
    assert models.implied_probs(None, 3.4, 3.8) is None


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_matches_python():
    script = f"const m=require({json.dumps(str(MATHS_JS))});" \
             f"console.log(JSON.stringify({json.dumps(CASES)}.map(c=>m.markets(...c))))"
    js = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for (lh, la, rho), j in zip(CASES, js):
        py = models.markets(lh, la, rho)
        for k in ("home", "draw", "away", "btts_yes", "home_cs", "away_cs"):
            assert j[k]["p"] == pytest.approx(py[k]["p"], abs=1e-9), (k, lh, la, rho)
            assert j[k]["american"] == py[k]["american"]
        for line in ("0.5", "1.5", "2.5", "3.5", "4.5"):
            assert j["over"][line]["p"] == pytest.approx(py["over"][line]["p"], abs=1e-9)
        assert np.allclose(j["matrix"], py["matrix"], atol=1e-9)
        assert [s["score"] for s in j["top_scores"][:3]] == [s["score"] for s in py["top_scores"][:3]]
