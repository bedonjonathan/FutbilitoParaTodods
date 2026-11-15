import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def league():
    """A small made-up league (no network): 10 teams, three seasons, known strengths."""
    rng = np.random.default_rng(7)
    teams = [f"T{i}" for i in range(10)]
    strength = {t: 1.6 - 0.12 * i for i, t in enumerate(teams)}   # T0 strongest
    rows = []
    for s, season in enumerate(["2425", "2526", "2627"]):
        day = pd.Timestamp(f"{2024 + s}-08-10")
        for rnd in range(2):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    hg = rng.poisson(1.25 * strength[h] / strength[a])
                    ag = rng.poisson(1.0 * strength[a] / strength[h])
                    rows.append(dict(date=day, season=season, home=h, away=a, hg=hg, ag=ag,
                                     hxg=np.nan, axg=np.nan, hs=10 + hg, hst=3 + hg, hc=5,
                                     ast=3 + ag, ac=4, odds_h=2.0, odds_d=3.4, odds_a=3.8, div="X"))
                    rows[-1]["as"] = 9 + ag
                    day += pd.Timedelta(days=1)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
