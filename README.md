# Futbolito Analytics ⚽

A local web app that pulls **free** football data automatically, shows it with charts, and
predicts matches using your spreadsheet's Poisson method plus three stronger models,
including a machine-learning model built on each team's **last 5 games**.

## Run it

```bash
pip install -r requirements.txt
python run.py            # opens http://localhost:8000
```

Click **Refresh data** at any time to download the latest results, odds and player stats.
Data is cached in `data/cache/`. The current season also refreshes itself if the cache is
more than 6 hours old, and past seasons are downloaded once.

## Data sources (free, no API key)

| Source | What we use | Leagues |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/data.php) | Results, shots, shots on target, corners, cards, **xG** (2026/27), average bookmaker odds, upcoming fixtures with odds | Premier League, Championship, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga. The current season plus 3 back. |
| [Fantasy Premier League API](https://fantasy.premierleague.com/api/bootstrap-static/) | Every PL player's goals, assists, xG, xA, minutes, form, injuries, per-gameweek history, and the PL fixture list | Premier League |

Optional for later: [football-data.org](https://www.football-data.org/) (free key) for live
scores and more competitions.

## Tabs

* **Overview**: league table with All / Home / Away splits (your sheet's two tables), xG,
  last-5 form, a team strength map (attack vs defence), Elo ranking and goals-per-week trend.
* **Teams**: rolling (3/5/8/10-game) goals and xG for and against, goal difference per match,
  points vs title/survival pace, and a full match log with shots and corners.
* **Match Predictor**: pick any two teams and a model. Shows expected goals, 1X2
  probabilities with fair decimal and American odds, a correct-score heat-map, over/under
  0.5 to 4.5, BTTS, clean sheets, every model side by side, last-5 form, head-to-head, and
  anytime-goalscorer chances (PL).
* **Upcoming**: the next fixtures, auto-predicted. When bookmaker odds are published, it
  shows the model's **edge** over the market.
* **Players** (PL): sortable and filterable player table, a goals vs xG finishing chart, and
  per-gameweek xG/xA for any player.
* **Model Lab**: a walk-forward backtest of every model against the bookmakers (accuracy,
  log-loss, RPS, Brier), a calibration chart, and what the ML model learned.

## Models

| Key | Model | Idea |
|---|---|---|
| `sheet` | Your spreadsheet | Season home/away attack × defence × league average → Poisson grid |
| `form5` | Last-5 form (+xG) | Same, from the last 5 games, goals blended 50/50 with xG, shrunk toward average |
| `dc` | Dixon-Coles | Maximum-likelihood team ratings over 3+ seasons, time-decayed, with a low-score correction |
| `ml` | ML Poisson | Poisson regressions on last-5 goals, shots on target, points and Elo, trained on every past match |
| `blend` | Ensemble | Geometric mean of the models' expected goals |

### Spreadsheet fixes carried over
* `B9` (Home Wins %) skipped the 1-0 cell (`G40`).
* `B18` (Under 3.5) skipped the 0-3 cell (`F43`).
* American odds used `(1/p − 1)·100`, which is only correct for underdogs. Favourites are now negative.
* Correct-score labels such as `1-1` had been turned into dates by Excel.
* The hard-coded 20-team `IFS` lookups are now automatic, so promoted teams just work.

## Project layout

```
backend/
  sources.py    downloads + caching (football-data.co.uk, FPL)
  analytics.py  tables, rolling form, Elo, pre-match ML features
  models.py     sheet / form5 / Dixon-Coles / ML models + market maths
  engine.py     fits models per league, walk-forward backtest
  app.py        FastAPI JSON API (/api/...) + serves the UI
frontend/       index.html, styles.css, app.js (Chart.js)
```

## Ideas for next steps
* **AI analyst**: send a match's stats to Claude and get a written preview, or a chat box:
  "why does the model like Brighton?"
* **More signals**: lineups and injuries feeding into the ML features, rest days, referee
  card rates, corners and cards markets.
* **Gradient boosting**: try gradient-boosted models and tune the decay/form window using the Model Lab.
* **History**: store predictions in SQLite to track real-world accuracy and profit/loss over time.
* **Scheduling**: refresh automatically every morning.
