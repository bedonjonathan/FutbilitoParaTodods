"""Choose the Dixon-Coles time-decay (xi) by walk-forward log-loss.

    python -m backend.tune                       # Premier League + three others
    python -m backend.tune --leagues E0 --xi 0.001 0.002 0.004

Not part of the weekly build: run it by hand once or twice a season and copy the
best value into models.DC_XI.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import models, sources


def dc_log_loss(hist, eval_seasons, xi: float) -> tuple[float, int]:
    ev = hist[hist.season.isin(eval_seasons)]
    weeks = ev.date.dt.to_period("W")
    ll = []
    for wk in sorted(weeks.unique()):
        idx = ev.index[weeks == wk]
        start = ev.loc[idx, "date"].min()
        train = hist[hist.date < start]
        if len(train) < 200:
            continue
        dc = models.DixonColes(xi=xi).fit(train, ref_date=start)
        for i in idx:
            r = hist.loc[i]
            lam = dc.lambdas(r.home, r.away)
            if lam is None:
                continue
            mk = models.markets(*lam, rho=dc.rho)
            p = [mk["home"]["p"], mk["draw"]["p"], mk["away"]["p"]]
            ll.append(-np.log(max(p[0 if r.hg > r.ag else 1 if r.hg == r.ag else 2], 1e-9)))
    return float(np.mean(ll)), len(ll)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", default=["E0", "SP1", "D1", "I1"])
    ap.add_argument("--xi", nargs="*", type=float, default=[0.0005, 0.001, 0.0019, 0.003, 0.0045, 0.0065])
    a = ap.parse_args(argv)
    totals = {x: [] for x in a.xi}
    for div in a.leagues:
        season = sources.active_season(div)
        hist = sources.load_history(div, season, n_prev=3).reset_index(drop=True)
        seasons = [sources.previous_seasons(season, 1)[0], season]
        row = []
        for x in a.xi:
            ll, n = dc_log_loss(hist, seasons, x)
            totals[x].append(ll)
            row.append(f"{x:.4f}: {ll:.4f}")
        print(f"{div} (n={n})  " + "  ".join(row))
    best = min(totals, key=lambda x: np.mean(totals[x]))
    print("average  " + "  ".join(f"{x:.4f}: {np.mean(v):.4f}" for x, v in totals.items()))
    print(f"best xi = {best}  (half-life {np.log(2) / best / 365:.1f} years); current models.DC_XI = {models.DC_XI}")


if __name__ == "__main__":
    main()
