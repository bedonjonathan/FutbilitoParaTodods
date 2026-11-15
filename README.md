# Futbolito Analytics ⚽

A website that pulls **free** football data automatically, shows it with charts, and
predicts matches using your spreadsheet's Poisson method plus four stronger models,
including machine-learning models built on each team's **last 5 games**.

**https://bedonjonathan.github.io/FutbilitoParaTodods/**

> For information and entertainment only, not betting advice. In the site's own tests the
> bookmakers' odds are more accurate than every model here.

## How it runs: no commands needed

GitHub Pages can only serve static files, and the football data sites don't allow browser
pages to fetch from them directly. So a **GitHub Action** (`.github/workflows/pages.yml`)
runs on GitHub's computers **every Thursday** (12:43 UTC), the day before the match week
starts. It runs the tests, downloads the latest data, fits every model, writes the results
as JSON files and publishes the site. Your browser just reads those files; the predictor
maths runs in the page.

* **Refresh data** (in the app) loads the newest published data.
* **Update now ↗** opens the workflow on GitHub; click **Run workflow** to rebuild any time
  (about 20 minutes).
* To change the schedule, edit the `cron:` line in `.github/workflows/pages.yml`.

### One-time setup (already done)
**Settings → Pages → Build and deployment → Source: "GitHub Actions"**.

### Optional: run it locally
```bash
pip install -r requirements.txt
python run.py --leagues E0     # build Premier League data and open http://localhost:8000
```

## Data sources (free, no API key)

| Source | What we use | Leagues |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/data.php) | Results, shots, shots on target, corners, cards, **xG** (2026/27), average bookmaker odds, upcoming fixtures with odds | Premier League, Championship, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga. Current and previous season are published; models train on the 3 seasons before. |
| Fantasy Premier League API | Every PL player's goals, assists, xG, xA, minutes, form, injuries, penalty takers, per-gameweek history, and the PL fixture list | Premier League |

Both are free feeds meant for personal use, and the FPL API is unofficial. This site is
not affiliated with either. If one of them changes or blocks automated access, that part
of the site will stop updating (see "When something goes wrong").

## Tabs

* **Overview**: league table with All / Home / Away splits (your sheet's two tables), xG,
  last-5 form, a team strength map (attack vs defence), Elo ranking and goals-per-week trend.
* **Teams**: rolling (3/5/8/10-game) goals and xG for and against, goal difference per match,
  points vs title/survival pace, and a full match log with shots and corners.
* **Match Predictor**: pick any two teams and a model. Shows expected goals, 1X2
  probabilities with fair decimal and American odds, a correct-score heat-map, over/under
  0.5 to 4.5, BTTS, clean sheets, every model side by side, last-5 form, head-to-head, and
  anytime-goalscorer chances (PL).
* **Upcoming**: the next fixtures, auto-predicted. When bookmaker odds are published it
  shows the **gap vs market**: a disagreement to look into, not a tip.
* **Players** (PL): sortable and filterable player table, a goals vs xG finishing chart, and
  per-gameweek xG/xA for any player.
* **Model Lab**: a walk-forward backtest of every model against the bookmakers over the
  last two seasons (accuracy, log-loss, RPS, Brier), a calibration chart, the blend's
  current weights, and what the ML model learned.

## Models

| Key | Model | Idea |
|---|---|---|
| `sheet` | Your spreadsheet | Season home/away attack × defence × league average → Poisson grid |
| `form5` | Last-5 form (+xG) | Same, from the last 5 games, goals blended 50/50 with xG, shrunk toward average |
| `dc` | Dixon-Coles | Maximum-likelihood team ratings over 3+ seasons, time-decayed, with a low-score correction |
| `ml` | ML Poisson | Poisson regressions on last-5 goals, shots on target, points and Elo |
| `gbm` | Gradient boosting | Boosted trees on the same signals plus last-5 xG and days of rest |
| `blend` | Accuracy-weighted blend | Weighted mean of the models' expected goals; weights come from each model's log-loss on matches already played |

Goalscorer chances use each player's xG per 90 × minutes over his last 5 gameweeks ×
fitness, as a share of his team's expected goals.

`python -m backend.tune` re-checks the Dixon-Coles time-decay setting against four
leagues; run it by hand once or twice a season.

### Spreadsheet fixes carried over
* `B9` (Home Wins %) skipped the 1-0 cell (`G40`).
* `B18` (Under 3.5) skipped the 0-3 cell (`F43`).
* American odds used `(1/p − 1)·100`, which is only correct for underdogs. Favourites are now negative.
* Correct-score labels such as `1-1` had been turned into dates by Excel.
* The hard-coded 20-team `IFS` lookups are now automatic, so promoted teams just work.

## When something goes wrong

The build is designed so a bad Thursday never replaces a working site with a broken one:

* **Tests fail** → nothing is published; last week's site stays up. GitHub emails you.
* **A league's download fails** → that league is rebuilt from the files saved last time,
  or last week's page is re-published. The page shows a "Heads up" banner, and the run on
  the Actions tab lists a warning.
* **No Premier League data at all** → the build stops and nothing is published.
* **A team name doesn't match** between the two sources (usually a promoted club in
  August) → its fixtures are skipped and the Actions run shows a warning naming the club.
  Add it to `FPL_TO_FDUK` in `backend/sources.py`.
* **Summer break** → each league keeps showing last season until the new one has matches.
* The header shows the date of the latest result in the data ("results through …") and
  when the site was built, so stale data is visible.

GitHub switches off scheduled workflows after 60 days without repository activity. The
workflow re-enables itself on every run to prevent that; if updates ever stop anyway, open
the Actions tab and click **Enable workflow**.

## Security notes

* No third-party requests: Chart.js and the Inter font are served from this site
  (`frontend/vendor/`), and a Content-Security-Policy blocks anything else.
* Python packages are pinned to exact versions and GitHub Actions to exact commits.
  Dependabot proposes updates monthly; the tests run on each proposal before you merge.
* The workflow's permissions are read-only except for publishing Pages.
* No logins, cookies, analytics or user data. The only thing stored in the browser is
  the light/dark theme choice.

## Project layout

```
backend/
  sources.py    downloads + caching, season roll-over, team-name matching
  analytics.py  tables, rolling form, Elo, pre-match ML features
  models.py     sheet / form5 / Dixon-Coles / ML / gradient-boosting models + market maths
  engine.py     fits models per league, accuracy-weighted blend, walk-forward backtest
  export.py     builds the static site: writes site/data/*.json + copies the UI
  tune.py       optional: re-tune the Dixon-Coles time decay
frontend/       index.html, styles.css, app.js, maths.js (odds maths), vendor/ (Chart.js, Inter)
tests/          pytest suite, incl. a check that the Python and JavaScript maths agree
.github/        weekly build + deploy workflow, Dependabot config
```

Run the tests with `pip install -r requirements-dev.txt && python -m pytest`.

## Ideas for next steps
* **AI analyst**: send a match's stats to Claude and get a written preview, or a chat box:
  "why does the model like Brighton?"
* **More signals**: confirmed lineups, referee card rates, corners and cards markets.
* **History**: store each week's predictions to track real-world accuracy over time.
