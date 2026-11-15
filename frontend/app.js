/* Futbolito Analytics – front end (vanilla JS + Chart.js) */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const state = { div: "E0", season: null, overview: null, players: null, charts: {}, tab: "overview", tableVenue: "all" };

// ------------------------------------------------------------------ helpers
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (p, d = 0) => (p == null ? "–" : (p * 100).toFixed(d) + "%");
const f2 = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const sign = (x, d = 2) => (x == null ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d));
const amer = (a) => (a == null ? "–" : a > 0 ? "+" + a : String(a));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const formChips = (s) => `<span class="form">${[...(s || "")].map((r) => `<span class="chip ${r}" title="${r === "W" ? "Win" : r === "D" ? "Draw" : "Loss"}">${r}</span>`).join("")}</span>`;

// All data is static JSON produced by backend/export.py (run by GitHub Actions).
const jsonCache = {};
async function getJSON(path, fresh = false) {
  const v = fresh ? Date.now() : state.meta?.generated || Date.now();
  if (!fresh && jsonCache[path]) return jsonCache[path];
  const r = await fetch(`data/${path}?v=${v}`, { cache: fresh ? "no-store" : "default" });
  if (!r.ok) throw new Error(r.status === 404 ? `No data published yet for ${path}` : r.statusText);
  return (jsonCache[path] = await r.json());
}
const L = () => state.league;

function workflowUrl() {
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const repo = location.pathname.split("/").filter(Boolean)[0];
  return m && repo ? `https://github.com/${m[1]}/${repo}/actions/workflows/pages.yml` : null;
}
function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}
function stamp(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return `Updated ${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

// ------------------------------------------------------------------ chart base
function chartDefaults() {
  Chart.defaults.font.family = '"Inter Variable", Inter, system-ui, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = css("--text-secondary");
  Chart.defaults.borderColor = css("--grid");
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.boxHeight = 8;
  Chart.defaults.plugins.tooltip.backgroundColor = css("--text-primary");
  Chart.defaults.plugins.tooltip.titleColor = css("--surface-1");
  Chart.defaults.plugins.tooltip.bodyColor = css("--surface-1");
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
  Chart.defaults.elements.line.borderWidth = 2;
  Chart.defaults.elements.line.tension = 0.25;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 5;
  Chart.defaults.elements.bar.borderRadius = 4;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.interaction = { mode: "index", intersect: false };
}
function draw(id, config) {
  if (state.charts[id]) state.charts[id].destroy();
  const el = document.getElementById(id);
  if (!el) return;
  state.charts[id] = new Chart(el, config);
  return state.charts[id];
}
const gridScale = (extra = {}) => ({ grid: { color: css("--grid"), drawTicks: false }, border: { display: false }, ticks: { padding: 6 }, ...extra });
const noGrid = (extra = {}) => ({ grid: { display: false }, border: { color: css("--border") }, ticks: { padding: 4 }, ...extra });

// ------------------------------------------------------------------ boot
async function boot() {
  chartDefaults();
  const meta = await getJSON("index.json", true);
  state.meta = meta;
  const wf = workflowUrl();
  if (wf) { $("#updateNow").href = wf; $("#updateNow").classList.remove("hidden"); }
  $("#league").innerHTML = meta.leagues.map((l) => `<option value="${l.code}">${esc(l.name)} · ${esc(l.country)}</option>`).join("");
  $("#season").innerHTML = meta.seasons.map((s) => `<option value="${s.code}">${s.label}</option>`).join("");
  $("#modelSelect").innerHTML = Object.entries(meta.models).map(([k, v]) => `<option value="${k}" ${k === "blend" ? "selected" : ""}>${esc(v)}</option>`).join("");
  state.season = meta.current_season;

  $("#league").onchange = (e) => { state.div = e.target.value; loadLeague(); };
  $("#season").onchange = (e) => { state.season = e.target.value; loadLeague(); };
  $("#refresh").onclick = refresh;
  $("#themeToggle").onclick = toggleTheme;
  const tabs = $$(".tab");
  tabs.forEach((b, i) => {
    b.onclick = () => showTab(b.dataset.tab);
    b.onkeydown = (e) => {   // arrow keys move between tabs, as screen-reader users expect
      const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      const to = e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : step ? (i + step + tabs.length) % tabs.length : null;
      if (to === null) return;
      e.preventDefault(); tabs[to].focus(); showTab(tabs[to].dataset.tab);
    };
  });
  $$("#tableVenue button").forEach((b) => (b.onclick = () => {
    $$("#tableVenue button").forEach((x) => x.classList.toggle("on", x === b));
    state.tableVenue = b.dataset.v; renderTable();
  }));
  $("#teamSelect").onchange = loadTeam;
  $("#teamWindow").onchange = loadTeam;
  ["homeSelect", "awaySelect"].forEach((id) => ($("#" + id).onchange = loadPrediction));
  $("#modelSelect").onchange = () => state.prediction && renderPrediction(state.prediction);
  $("#swap").onclick = () => { const h = $("#homeSelect").value; $("#homeSelect").value = $("#awaySelect").value; $("#awaySelect").value = h; loadPrediction(); };
  ["pTeam", "pPos", "pMin", "pSearch"].forEach((id) => ($("#" + id).oninput = renderPlayers));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerenderAll);

  const saved = safeGet("theme"); if (saved) document.documentElement.dataset.theme = saved;
  await loadLeague();
}
function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch {} }

function toggleTheme() {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  safeSet("theme", document.documentElement.dataset.theme);
  rerenderAll();
}
function rerenderAll() {
  chartDefaults();
  if (state.overview) renderOverview(state.overview);
  if (state.team) renderTeam(state.team);
  if (state.prediction) renderPrediction(state.prediction);
  if (state.players) renderPlayers();
  if (state.playerHist) renderPlayerChart(...state.playerHist);
  if (state.backtest) renderBacktest(state.backtest);
}

function showTab(tab) {
  state.tab = tab;
  $$(".tab").forEach((b) => {
    const on = b.dataset.tab === tab;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
    b.tabIndex = on ? 0 : -1;
  });
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tab));
  if (tab === "team" && !state.team) loadTeam();
  if (tab === "predict" && !state.prediction) loadPrediction();
  if (tab === "fixtures" && !state.fixtures) loadFixtures();
  if (tab === "players" && !state.players) loadPlayers();
  if (tab === "lab" && !state.backtest) loadBacktest();
}

async function refresh() {
  const b = $("#refresh"); b.disabled = true; b.classList.add("busy");
  try {
    const prev = state.meta.generated;
    const meta = await getJSON("index.json", true);
    if (meta.generated === prev) {
      toast("You already have the latest data. It updates automatically every Thursday" + (workflowUrl() ? " — or use “Update now”." : "."), 6000);
    } else {
      for (const k in jsonCache) delete jsonCache[k];
      state.meta = meta; state.players = null; state.playerHistAll = null;
      jsonCache["index.json"] = meta;
      await loadLeague();
      toast("Loaded the newest data");
    }
  } catch (e) { toast("Refresh failed: " + e.message, 6000); }
  finally { b.disabled = false; b.classList.remove("busy"); }
}

async function loadLeague() {
  $("#dataStamp").textContent = "Loading…";
  Object.assign(state, { team: null, prediction: null, fixtures: null, backtest: null });
  try {
    let key = `${state.div}_${state.season}`;
    if (!state.meta.available.includes(key)) {
      // e.g. a league whose new season hasn't kicked off yet: show its latest published season
      const fallback = state.meta.default_season?.[state.div];
      if (!fallback) throw new Error("No data is published for this league yet.");
      state.season = fallback; $("#season").value = fallback; key = `${state.div}_${fallback}`;
    }
    const o = await getJSON(`${key}.json`);
    state.league = o; state.overview = o;
    const through = o.data_through ? new Date(o.data_through + "T12:00:00").toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" }) : "–";
    $("#dataStamp").textContent = `${o.league.name} ${o.season_label} · ${o.stats.matches || 0} matches · results through ${through} · ${stamp(state.meta.generated).replace("Updated", "built")}`;
    const kept = (state.meta.kept_from_previous_build || []).includes(key);
    const problems = [];
    if (o.stale || kept) problems.push("The last update couldn't download fresh results for this league, so this page shows data saved from an earlier update.");
    if (o.league.players && (state.meta.warnings || []).some((w) => w.includes("FPL"))) problems.push("Player data could not be fully refreshed in the last update.");
    $("#banner").classList.toggle("hidden", !problems.length);
    $("#banner").innerHTML = problems.length ? `<b>Heads up:</b> ${problems.map(esc).join(" ")}` : "";
    const opts = o.teams.map((t) => `<option>${esc(t)}</option>`).join("");
    const keepTeam = $("#teamSelect").value;
    $("#teamSelect").innerHTML = opts;
    if (o.teams.includes(keepTeam)) $("#teamSelect").value = keepTeam; else $("#teamSelect").value = o.table[0]?.team;
    $("#homeSelect").innerHTML = opts; $("#awaySelect").innerHTML = opts;
    $("#homeSelect").value = o.table[0]?.team; $("#awaySelect").value = o.table[1]?.team;
    const playersOk = o.league.players && state.season === state.meta.default_season?.[state.div] && state.meta.players;
    $("#playersUnavailable").classList.toggle("hidden", playersOk);
    $("#playersBody").classList.toggle("hidden", !playersOk);
    renderOverview(o);
    showTab(state.tab);
  } catch (e) {
    $("#dataStamp").textContent = "Could not load data";
    toast(e.message, 6000);
  }
}

// ------------------------------------------------------------------ overview
function kpi(label, value, sub = "") {
  return `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}
function renderOverview(o) {
  const s = o.stats;
  $("#kpis").innerHTML = s.matches ? [
    kpi("Goals per game", f2(s.goals_per_game), `${f2(s.home_goals_pg)} home · ${f2(s.away_goals_pg)} away`),
    kpi("Home wins", pct(s.home_win), `Draws ${pct(s.draw)} · Away ${pct(s.away_win)}`),
    kpi("Over 2.5 goals", pct(s.over25)),
    kpi("Both teams score", pct(s.btts)),
    kpi("Home advantage", "×" + f2(o.dc_home_adv), "goal multiplier (Dixon-Coles)"),
  ].join("") : kpi("Matches", "0", "Season hasn't started");
  renderTable();

  // strength map – one series, every team same colour, labels in tooltip + selective direct labels
  const r = o.dc_ratings;
  const pts = r.map((t) => ({ x: t.attack, y: 1 / t.defence, team: t.team }));
  const ranked = [...pts].sort((a, b) => b.x * b.y - a.x * a.y);
  const labelSet = new Set([...ranked.slice(0, 4), ...ranked.slice(-3)].map((p) => p.team));
  draw("strengthChart", {
    type: "scatter",
    data: { datasets: [{ data: pts, backgroundColor: css("--series-1"), borderColor: css("--surface-1"), borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 }] },
    options: {
      interaction: { mode: "nearest", intersect: true },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `${c.raw.team}: attack ×${f2(c.raw.x)}, defence ×${f2(1 / c.raw.y)} goals allowed` } },
      },
      scales: {
        x: gridScale({ title: { display: true, text: "Attack (× league average goals)" } }),
        y: gridScale({ title: { display: true, text: "Defence (higher = concedes less)" } }),
      },
      layout: { padding: { right: 30, top: 10 } },
    },
    plugins: [labelPlugin((p) => (labelSet.has(p.team) ? p.team : null))],
  });

  // Elo – horizontal bars, one colour
  draw("eloChart", {
    type: "bar",
    data: { labels: o.elo.map((e) => e.team), datasets: [{ data: o.elo.map((e) => e.elo), backgroundColor: css("--series-1"), barPercentage: 0.7, categoryPercentage: 0.9 }] },
    options: {
      indexAxis: "y",
      interaction: { mode: "nearest", axis: "y", intersect: false },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `Elo ${Math.round(c.raw)}` } } },
      scales: { x: gridScale({ min: Math.floor((Math.min(...o.elo.map((e) => e.elo)) - 40) / 50) * 50 }), y: noGrid({ ticks: { autoSkip: false, font: { size: 11 } } }) },
    },
  });

  // trend
  const avg = s.goals_per_game;
  draw("trendChart", {
    type: "line",
    data: {
      labels: o.trend.map((t) => t.week),
      datasets: [
        { label: "Goals per game", data: o.trend.map((t) => t.gpg), borderColor: css("--series-1"), backgroundColor: css("--series-1"), pointRadius: 3 },
        { label: "Season average", data: o.trend.map(() => avg), borderColor: css("--neutral"), borderWidth: 1, pointRadius: 0, pointHoverRadius: 0 },
      ],
    },
    options: { plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${f2(c.raw)}` } } }, scales: { x: noGrid({ ticks: { maxTicksLimit: 10 } }), y: gridScale({ beginAtZero: true }) } },
  });
}

// draws text next to selected scatter points
function labelPlugin(fn) {
  return {
    id: "pointLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      ctx.save();
      ctx.font = '500 11px "Inter Variable", Inter, system-ui';
      ctx.fillStyle = css("--text-secondary");
      chart.data.datasets.forEach((ds, i) => {
        chart.getDatasetMeta(i).data.forEach((el, j) => {
          const t = fn(ds.data[j]);
          if (t) ctx.fillText(t, el.x + 9, el.y + 4);
        });
      });
      ctx.restore();
    },
  };
}

function renderTable() {
  const o = state.overview; if (!o) return;
  const rows = state.tableVenue === "home" ? o.home_table : state.tableVenue === "away" ? o.away_table : o.table;
  const xg = o.stats.has_xg;
  const head = `<thead><tr><th class="l">#</th><th class="l">Team</th><th>GP</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th>GF/GP</th><th>GA/GP</th>${xg ? "<th>xGF</th><th>xGA</th><th>xGD</th>" : ""}<th class="l">Last 5</th></tr></thead>`;
  const body = rows.map((r) => `<tr class="clickable" data-team="${esc(r.team)}"><td class="l pos">${r.pos}</td><td class="l team-cell">${esc(r.team)}</td><td>${r.M}</td><td>${r.W}</td><td>${r.D}</td><td>${r.L}</td><td>${r.GF}</td><td>${r.GA}</td><td>${sign(r.GD, 0)}</td><td><b>${r.PTS}</b></td><td>${f2(r.GF_pg)}</td><td>${f2(r.GA_pg)}</td>${xg ? `<td>${f2(r.xGF, 1)}</td><td>${f2(r.xGA, 1)}</td><td>${sign(r.xGD, 1)}</td>` : ""}<td class="l">${formChips(r.form)}</td></tr>`).join("");
  $("#leagueTable").innerHTML = head + "<tbody>" + body + "</tbody>";
  $$("#leagueTable tr.clickable").forEach((tr) => (tr.onclick = () => { $("#teamSelect").value = tr.dataset.team; state.team = null; showTab("team"); }));
}

// ------------------------------------------------------------------ team
async function loadTeam() {
  const team = $("#teamSelect").value; if (!team) return;
  const w = +$("#teamWindow").value;
  const raw = L().series[team] || [];
  const roll = (k) => rolling(raw.map((m) => m[k]), w);
  const cols = ["gf", "ga", "xgf", "xga", "pts", "sotf", "sota"];
  const rolled = Object.fromEntries(cols.map((k) => [k, roll(k)]));
  const series = raw.map((m, i) => ({ ...m, ...Object.fromEntries(cols.map((k) => [k + "_roll", rolled[k][i]])) }));
  const lastW = raw.slice(-w);
  const avg = (k) => (lastW.length ? lastW.reduce((a, m) => a + (m[k] ?? 0), 0) / lastW.length : null);
  state.team = {
    team, series, dc: L().dc_ratings.find((r) => r.team === team), elo: L().elo.find((r) => r.team === team)?.elo,
    form: { gf_f: avg("gf"), ga_f: avg("ga"), pts_f: avg("pts") },
  };
  renderTeam(state.team);
}
function renderTeam(t) {
  const s = t.series, w = $("#teamWindow").value;
  $("#rollLbl").textContent = `(rolling ${w})`;
  const labels = s.map((m) => `${m.venue === "H" ? "vs" : "@"} ${m.opp}`);
  const last = s[s.length - 1] || {};
  const ppg = s.length ? last.cum_pts / s.length : 0;
  $("#teamKpis").innerHTML = [
    kpi("Points", last.cum_pts ?? 0, `${f2(ppg)} per game`),
    kpi(`Last ${w}: scored`, f2(t.form.gf_f), "goals per game"),
    kpi(`Last ${w}: conceded`, f2(t.form.ga_f), "goals per game"),
    kpi(`Last ${w}: points`, f2(t.form.pts_f), "per game"),
    kpi("Elo", t.elo ? Math.round(t.elo) : "–"),
    kpi("DC attack / defence", t.dc ? `×${f2(t.dc.attack)} / ×${f2(t.dc.defence)}` : "–", "vs league average"),
  ].join("");

  const tip = { callbacks: { title: (items) => `${s[items[0].dataIndex].date} · ${labels[items[0].dataIndex]} (${s[items[0].dataIndex].gf}-${s[items[0].dataIndex].ga})`, label: (c) => `${c.dataset.label}: ${f2(c.raw)}` } };
  const line = (label, data, color, extra = {}) => ({ label, data, borderColor: color, backgroundColor: color, pointRadius: 0, ...extra });
  const xScale = noGrid({ ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8, callback: (v, i) => s[i]?.matchday } , title: { display: true, text: "Matchday" } });

  draw("teamGoals", { type: "line", data: { labels, datasets: [
    line("Scored", s.map((m) => m.gf_roll), css("--series-1")), line("Conceded", s.map((m) => m.ga_roll), css("--series-2")),
  ] }, options: { plugins: { tooltip: tip }, scales: { x: xScale, y: gridScale({ beginAtZero: true }) } } });

  const hasXg = s.some((m) => m.xgf != null);
  $("#xgNote").textContent = hasXg ? "xG measures chance quality. If goals run well above xG, expect regression." : "This league/season doesn't publish xG in the free feed — showing shots on target instead.";
  draw("teamXg", { type: "line", data: { labels, datasets: hasXg ? [
    line("xG for", s.map((m) => m.xgf_roll), css("--series-1")), line("xG against", s.map((m) => m.xga_roll), css("--series-2")),
  ] : [
    line("Shots on target for", s.map((m) => m.sotf_roll), css("--series-1")), line("Shots on target against", s.map((m) => m.sota_roll), css("--series-2")),
  ] }, options: { plugins: { tooltip: tip }, scales: { x: xScale, y: gridScale({ beginAtZero: true }) } } });

  draw("teamGd", { type: "bar", data: { labels, datasets: [{ label: "Goal difference", data: s.map((m) => m.gf - m.ga),
    backgroundColor: s.map((m) => (m.gf - m.ga > 0 ? css("--div-pos") : m.gf - m.ga < 0 ? css("--div-neg") : css("--neutral"))), minBarLength: 3 }] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { title: tip.callbacks.title, label: (c) => `GD ${sign(c.raw, 0)}` } } }, scales: { x: xScale, y: gridScale({ ticks: { stepSize: 1 } }) } } });

  // points vs a "title pace" (2.3 ppg) and "survival pace" (1.0 ppg) reference
  draw("teamPts", { type: "line", data: { labels, datasets: [
    line(t.team, s.map((m) => m.cum_pts), css("--series-1"), { pointRadius: 2 }),
    line("Title pace (2.3/game)", s.map((_, i) => +(2.3 * (i + 1)).toFixed(1)), css("--neutral"), { borderWidth: 1, borderDash: [4, 4] }),
    line("Survival pace (1.0/game)", s.map((_, i) => i + 1), css("--neutral"), { borderWidth: 1, borderDash: [2, 3] }),
  ] }, options: { plugins: { tooltip: tip }, scales: { x: xScale, y: gridScale({ beginAtZero: true }) } } });

  $("#teamLog").innerHTML = `<thead><tr><th class="l">Date</th><th class="l">Opponent</th><th class="l">Venue</th><th>Score</th><th class="l">Result</th><th>xG</th><th>xGA</th><th>Shots</th><th>On target</th><th>Corners</th><th>Pts total</th></tr></thead><tbody>` +
    [...s].reverse().map((m) => `<tr><td class="l">${m.date}</td><td class="l">${esc(m.opp)}</td><td class="l">${m.venue === "H" ? "Home" : "Away"}</td><td>${m.gf}-${m.ga}</td><td class="l">${formChips(m.res)}</td><td>${f2(m.xgf)}</td><td>${f2(m.xga)}</td><td>${m.sf ?? "–"}-${m.sa ?? "–"}</td><td>${m.sotf ?? "–"}-${m.sota ?? "–"}</td><td>${m.cf ?? "–"}-${m.ca ?? "–"}</td><td>${m.cum_pts}</td></tr>`).join("") + "</tbody>";
}

// ------------------------------------------------------------------ predictor
async function loadPrediction() {
  if (loadPrediction.busy) return;
  const h = $("#homeSelect").value, a = $("#awaySelect").value;
  if (!h || !a) return;
  if (h === a) { $("#predictOut").innerHTML = `<div class="card"><p>Pick two different teams.</p></div>`; return; }
  $("#predictOut").innerHTML = `<div class="card loading">Crunching…</div>`;
  const lg = L(), lams = lg.pairs[`${h}|${a}`] || {};
  const mods = {};
  for (const k of Object.keys(state.meta.models)) {
    const l = lams[k];
    mods[k] = l ? markets(l[0], l[1], k === "dc" || k === "blend" ? lg.rho : 0) : null;
  }
  const fx = (lg.fixtures || []).find((f) => f.home === h && f.away === a && f.odds);
  let scorers = null;
  if (lg.scorers && mods.blend) {
    const sc = (team, lam) => (lg.scorers[team] || []).map((p) => ({ ...p, p_score: 1 - Math.exp(-lam * p.share * p.availability) }))
      .sort((x, y) => y.p_score - x.p_score).slice(0, 8);
    scorers = { home: sc(h, mods.blend.lambda_home), away: sc(a, mods.blend.lambda_away) };
  }
  const eloOf = (t) => lg.elo.find((r) => r.team === t)?.elo;
  state.prediction = {
    home: h, away: a, models: mods, model_names: state.meta.models,
    h2h: lg.h2h[`${h}|${a}`] || lg.h2h[`${a}|${h}`] || [],
    form: { home: (lg.series[h] || []).slice(-5), away: (lg.series[a] || []).slice(-5) },
    elo: { home: eloOf(h), away: eloOf(a) },
    market: fx ? { odds: fx.odds, implied: fx.market_p } : null,
    scorers,
  };
  renderPrediction(state.prediction);
}

function renderPrediction(P) {
  let key = $("#modelSelect").value;
  if (!P.models[key]) key = Object.keys(P.models).find((k) => P.models[k]) || key;
  const m = P.models[key];
  if (!m) { $("#predictOut").innerHTML = `<div class="card">Not enough matches yet for this model.</div>`; return; }
  const H = css("--series-1"), A = css("--series-2"), D = css("--neutral");

  // heatmap (sequential blue)
  const max = Math.max(...m.matrix.flat());
  const heatRows = m.matrix.map((row, h) => `<tr><th class="axis">${h}</th>${row.map((p, a) => {
    const t = Math.sqrt(p / max);
    const bg = `color-mix(in oklab, var(--heat-1) ${Math.round(t * 100)}%, var(--heat-0))`;
    const fg = t > 0.55 ? (document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches) ? "#0b0b0b" : "#fff") : "var(--text-secondary)";
    return `<td style="background:${bg};color:${fg}" title="${esc(P.home)} ${h} – ${a} ${esc(P.away)}: ${pct(p, 1)}">${(p * 100).toFixed(1)}</td>`;
  }).join("")}</tr>`).join("");

  const ou = ["0.5", "1.5", "2.5", "3.5", "4.5"].map((t) => `<tr><td class="l">${t}</td><td>${pct(m.over[t].p, 1)}</td><td>${f2(m.over[t].decimal)}</td><td>${amer(m.over[t].american)}</td><td>${pct(m.under[t].p, 1)}</td><td>${f2(m.under[t].decimal)}</td><td>${amer(m.under[t].american)}</td></tr>`).join("");

  const order = ["sheet", "form5", "dc", "ml", "gbm", "blend"];
  const cmp = order.filter((k) => P.models[k]).map((k) => {
    const x = P.models[k];
    return `<tr class="${k === key ? "best" : ""}"><td class="l">${esc(P.model_names[k])}</td><td>${f2(x.lambda_home)}</td><td>${f2(x.lambda_away)}</td><td>${pct(x.home.p, 1)}</td><td>${pct(x.draw.p, 1)}</td><td>${pct(x.away.p, 1)}</td><td>${pct(x.over["2.5"].p, 0)}</td><td>${pct(x.btts_yes.p, 0)}</td><td class="l">${x.top_scores[0].score}</td></tr>`;
  });
  let marketRow = "";
  if (P.market?.implied) {
    const q = P.market.implied;
    marketRow = `<tr><td class="l">Bookmakers (avg odds ${P.market.odds.map((o) => f2(o)).join(" / ")})</td><td>–</td><td>–</td><td>${pct(q[0], 1)}</td><td>${pct(q[1], 1)}</td><td>${pct(q[2], 1)}</td><td>–</td><td>–</td><td class="l">–</td></tr>`;
  }

  const formList = (rows) => rows.slice().reverse().map((r) => `<div class="row"><span>${r.date.slice(5)} ${r.venue === "H" ? "vs" : "@"} ${esc(r.opp)}</span><span>${r.gf}-${r.ga} ${formChips(r.res)}</span></div>`).join("") || "<span class='muted'>No matches yet</span>";
  const h2h = P.h2h.map((g) => `<tr><td class="l">${g.date}</td><td class="l">${esc(g.home)}</td><td>${g.hg}-${g.ag}</td><td class="l">${esc(g.away)}</td></tr>`).join("") || `<tr><td class="l muted">No meetings in the data window</td></tr>`;

  const scorerBlock = (rows, color) => rows.map((r) => `<div class="scorer" title="${esc(r.news || "")}"><span>${esc(r.name)}${r.pens ? ` <span class="tag" title="First-choice penalty taker">pens</span>` : ""}${r.availability < 1 ? ` <span class="muted">(${Math.round(r.availability * 100)}% fit)</span>` : ""}</span><span class="bar"><i style="width:${Math.min(100, r.p_score * 200)}%;background:${color}"></i></span><span class="num">${pct(r.p_score)}</span></div>`).join("");

  $("#predictOut").innerHTML = `
  <div class="card">
    <div class="match-hero">
      <div><div class="team"><span class="swatch" style="background:${H}"></span>${esc(P.home)}</div><div class="xg">${f2(m.lambda_home)}</div><div class="muted">expected goals · Elo ${P.elo.home ? Math.round(P.elo.home) : "–"}</div></div>
      <div class="vs">vs<br><span class="muted">${esc(P.model_names[key])}</span></div>
      <div><div class="team"><span class="swatch" style="background:${A}"></span>${esc(P.away)}</div><div class="xg">${f2(m.lambda_away)}</div><div class="muted">expected goals · Elo ${P.elo.away ? Math.round(P.elo.away) : "–"}</div></div>
    </div>
    <div class="prob-bar" role="img" aria-label="Home ${pct(m.home.p)} draw ${pct(m.draw.p)} away ${pct(m.away.p)}">
      <div style="flex:${m.home.p};background:${H}">${m.home.p > 0.08 ? pct(m.home.p) : ""}</div>
      <div style="flex:${m.draw.p};background:${D}">${m.draw.p > 0.08 ? pct(m.draw.p) : ""}</div>
      <div style="flex:${m.away.p};background:${A}">${m.away.p > 0.08 ? pct(m.away.p) : ""}</div>
    </div>
    <div class="prob-legend">
      <span><b>${esc(P.home)} win</b> ${pct(m.home.p, 1)} · fair ${f2(m.home.decimal)} (${amer(m.home.american)})</span>
      <span><b>Draw</b> ${pct(m.draw.p, 1)} · fair ${f2(m.draw.decimal)} (${amer(m.draw.american)})</span>
      <span><b>${esc(P.away)} win</b> ${pct(m.away.p, 1)} · fair ${f2(m.away.decimal)} (${amer(m.away.american)})</span>
    </div>
  </div>

  <div class="grid two" style="margin-top:16px">
    <div class="card">
      <h2>Correct score probability (%)</h2>
      <p class="hint">Rows = ${esc(P.home)} goals, columns = ${esc(P.away)} goals.</p>
      <div class="table-wrap"><table class="data heat" style="width:auto"><thead><tr><th class="axis">H \\ A</th>${[0,1,2,3,4,5,6].map((i) => `<th>${i}</th>`).join("")}</tr></thead><tbody>${heatRows}</tbody></table></div>
      <p class="note">Most likely: ${m.top_scores.slice(0, 5).map((s) => `<b>${s.score}</b> ${pct(s.p, 1)}`).join(" · ")}</p>
    </div>
    <div class="card">
      <h2>Goal markets</h2>
      <div class="table-wrap"><table class="data"><thead><tr><th class="l">Line</th><th>Over</th><th>Fair</th><th>US</th><th>Under</th><th>Fair</th><th>US</th></tr></thead><tbody>${ou}</tbody></table></div>
      <div class="kpis" style="margin-top:14px;margin-bottom:0">
        ${kpi("Both teams score", pct(m.btts_yes.p), `fair ${f2(m.btts_yes.decimal)} (${amer(m.btts_yes.american)})`)}
        ${kpi(`${esc(P.home)} clean sheet`, pct(m.home_cs.p))}
        ${kpi(`${esc(P.away)} clean sheet`, pct(m.away_cs.p))}
      </div>
    </div>
    <div class="card">
      <h2>Goals distribution</h2>
      <div class="chart short" style="height:220px"><canvas id="goalDist"></canvas></div>
    </div>
    <div class="card">
      <h2>Last 5 matches</h2>
      <div class="grid two inner">
        <div><h3>${esc(P.home)}</h3><div class="mini-form">${formList(P.form.home)}</div></div>
        <div><h3>${esc(P.away)}</h3><div class="mini-form">${formList(P.form.away)}</div></div>
      </div>
    </div>
    <div class="card span2">
      <h2>All models side by side</h2>
      <div class="table-wrap"><table class="data"><thead><tr><th class="l">Model</th><th>xG home</th><th>xG away</th><th>Home</th><th>Draw</th><th>Away</th><th>O2.5</th><th>BTTS</th><th class="l">Top score</th></tr></thead><tbody>${cmp.join("")}${marketRow}</tbody></table></div>
      <p class="note">Check the Model Lab tab to see which model has actually been most accurate this season.</p>
    </div>
    ${P.scorers ? `<div class="card span2"><h2>Anytime goalscorer chances <span class="muted">(team expected goals × player's xG rate × recent minutes × fitness)</span></h2>
      <div class="grid two inner"><div><h3>${esc(P.home)}</h3>${scorerBlock(P.scorers.home, H)}</div><div><h3>${esc(P.away)}</h3>${scorerBlock(P.scorers.away, A)}</div></div></div>` : ""}
    <div class="card span2">
      <h2>Head to head <span class="muted">(last 3 seasons)</span></h2>
      <div class="table-wrap"><table class="data"><tbody>${h2h}</tbody></table></div>
    </div>
  </div>`;

  draw("goalDist", { type: "bar", data: { labels: ["0", "1", "2", "3", "4", "5", "6"], datasets: [
    { label: P.home, data: m.home_goals_dist, backgroundColor: H }, { label: P.away, data: m.away_goals_dist, backgroundColor: A },
  ] }, options: { plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.raw, 1)}` } } }, scales: { x: noGrid({ title: { display: true, text: "Goals scored" } }), y: gridScale({ ticks: { callback: (v) => pct(v) } }) } } });
}

// ------------------------------------------------------------------ fixtures
async function loadFixtures() {
  state.fixtures = { fixtures: (L().fixtures || []).slice(0, 30) };
  renderFixtures(state.fixtures);
}
function renderFixtures(F) {
  if (!F.fixtures.length) { $("#fixturesOut").innerHTML = `<p class="muted">No upcoming fixtures in the free feed for this league right now (international break, or the feed only lists this weekend's games). Use the Match Predictor for any pairing.</p>`; return; }
  const H = css("--series-1"), A = css("--series-2"), D = css("--neutral");
  const anyOdds = F.fixtures.some((f) => f.edge);
  const rows = F.fixtures.map((f) => {
    const dt = f.kickoff ? new Date(f.kickoff) : null;
    const when = dt ? dt.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" }) + " " + dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
    const edge = f.edge ? f.edge.map((e) => `<span class="${Math.abs(e) >= 0.05 ? "best" : "edge-neg"}">${sign(e * 100, 1)}</span>`).join(" / ") : "–";
    return `<tr class="clickable" data-h="${esc(f.home)}" data-a="${esc(f.away)}"><td class="l">${f.round ? "GW" + f.round + " · " : ""}${when}</td><td class="l team-cell">${esc(f.home)}</td><td class="l team-cell">${esc(f.away)}</td>
      <td>${f2(f.lambda_home)} – ${f2(f.lambda_away)}</td>
      <td class="l"><div class="small-bar" title="Home ${pct(f.p[0])} · Draw ${pct(f.p[1])} · Away ${pct(f.p[2])}"><div style="flex:${f.p[0]};background:${H}"></div><div style="flex:${f.p[1]};background:${D}"></div><div style="flex:${f.p[2]};background:${A}"></div></div></td>
      <td>${pct(f.p[0])}</td><td>${pct(f.p[1])}</td><td>${pct(f.p[2])}</td><td>${pct(f.over25)}</td><td>${pct(f.btts)}</td><td>${f.top_score}</td>${anyOdds ? `<td>${edge}</td>` : ""}</tr>`;
  }).join("");
  $("#fixturesOut").innerHTML = `<div class="table-wrap"><table class="data"><thead><tr><th class="l">Kick-off</th><th class="l">Home</th><th class="l">Away</th><th>Exp. goals</th><th class="l"><span class="swatch" style="background:${H}"></span>H <span class="swatch" style="background:${D}"></span>D <span class="swatch" style="background:${A}"></span>A</th><th>Home</th><th>Draw</th><th>Away</th><th>O2.5</th><th>BTTS</th><th>Top score</th>${anyOdds ? '<th title="Model probability minus bookmaker probability, in percentage points. Bookmakers have been more accurate than the model, so treat gaps as questions, not tips.">Gap vs market H/D/A</th>' : ""}</tr></thead><tbody>${rows}</tbody></table></div>
  <p class="note">Click a fixture to open it in the Match Predictor.</p>`;
  $$("#fixturesOut tr.clickable").forEach((tr) => (tr.onclick = () => { $("#homeSelect").value = tr.dataset.h; $("#awaySelect").value = tr.dataset.a; showTab("predict"); loadPrediction(); }));
}

// ------------------------------------------------------------------ players
async function loadPlayers() {
  if (!(state.overview?.league.players)) return;
  try {
    state.players = await getJSON("players.json");
    $("#pTeam").innerHTML = `<option value="">All teams</option>` + state.players.teams.map((t) => `<option>${esc(t)}</option>`).join("");
    state.pSort = { key: "xg", asc: false };
    renderPlayers();
  } catch (e) { toast(e.message); }
}
const PCOLS = [
  ["name", "Player", "l"], ["team", "Team", "l"], ["pos", "Pos", "l"], ["minutes", "Min"], ["goals", "G"], ["xg", "xG"], ["g_minus_xg", "G−xG"],
  ["assists", "A"], ["xa", "xA"], ["xg90", "xG/90"], ["xa90", "xA/90"], ["form", "Form"], ["points", "FPL pts"], ["price", "£m"], ["selected", "Own %"],
];
function filteredPlayers() {
  const team = $("#pTeam").value, pos = $("#pPos").value, min = +$("#pMin").value || 0, q = $("#pSearch").value.toLowerCase();
  return state.players.players.filter((p) => (!team || p.team === team) && (!pos || p.pos === pos) && p.minutes >= min && (!q || p.full_name.toLowerCase().includes(q) || p.name.toLowerCase().includes(q)));
}
function renderPlayers() {
  if (!state.players) return;
  const rows = filteredPlayers();
  const { key, asc } = state.pSort;
  rows.sort((a, b) => (typeof a[key] === "string" ? a[key].localeCompare(b[key]) : a[key] - b[key]) * (asc ? 1 : -1));
  const fmt = (k, v) => (["xg", "xa", "g_minus_xg"].includes(k) ? (k === "g_minus_xg" ? sign(v, 1) : f2(v, 1)) : ["xg90", "xa90"].includes(k) ? f2(v) : k === "price" ? f2(v, 1) : v);
  $("#playersTable").innerHTML = `<thead><tr>${PCOLS.map(([k, l, c]) => `<th class="${c || ""} ${k === key ? "sorted" + (asc ? " asc" : "") : ""}" data-k="${k}">${l}</th>`).join("")}</tr></thead><tbody>` +
    rows.slice(0, 300).map((p) => `<tr class="clickable" data-id="${p.id}">${PCOLS.map(([k, , c]) => `<td class="${c || ""}">${k === "name" ? `<b>${esc(p.name)}</b>${p.status !== "a" ? ` <span class="pos-chip" title="${esc(p.news)}">${p.status === "i" ? "injured" : p.status === "d" ? "doubt" : p.status === "s" ? "suspended" : "out"}</span>` : ""}` : esc(fmt(k, p[k]))}</td>`).join("")}</tr>`).join("") + "</tbody>";
  $$("#playersTable th").forEach((th) => (th.onclick = () => { const k = th.dataset.k; state.pSort = { key: k, asc: state.pSort.key === k ? !state.pSort.asc : false }; renderPlayers(); }));
  $$("#playersTable tr.clickable").forEach((tr) => (tr.onclick = () => loadPlayer(+tr.dataset.id)));

  // scatter: xG vs goals (attackers with some output), one colour, diagonal reference
  const pts = rows.filter((p) => p.xg > 0.3 || p.goals > 0).map((p) => ({ x: p.xg, y: p.goals, p }));
  const maxv = Math.max(1, ...pts.map((d) => Math.max(d.x, d.y))) + 0.5;
  const top = new Set([...pts].sort((a, b) => b.y - a.y).slice(0, 5).concat([...pts].sort((a, b) => Math.abs(b.y - b.x) - Math.abs(a.y - a.x)).slice(0, 3)).map((d) => d.p.id));
  draw("xgScatter", {
    type: "scatter",
    data: { datasets: [
      { label: "Goals = xG", type: "line", data: [{ x: 0, y: 0 }, { x: maxv, y: maxv }], borderColor: css("--neutral"), borderWidth: 1, pointRadius: 0, pointHoverRadius: 0 },
      { label: "Players", data: pts, backgroundColor: css("--series-1"), borderColor: css("--surface-1"), borderWidth: 1.5, pointRadius: 5, pointHoverRadius: 7 },
    ] },
    options: {
      interaction: { mode: "nearest", intersect: true },
      onClick: (_, els) => { const e = els.find((x) => x.datasetIndex === 1); if (e) loadPlayer(pts[e.index].p.id); },
      plugins: { legend: { display: false }, tooltip: { filter: (i) => i.datasetIndex === 1, callbacks: { label: (c) => `${c.raw.p.name} (${c.raw.p.team}): ${c.raw.y} goals from ${f2(c.raw.x, 1)} xG` } } },
      scales: { x: gridScale({ min: 0, max: maxv, title: { display: true, text: "Expected goals (xG)" } }), y: gridScale({ min: 0, max: maxv, title: { display: true, text: "Goals" } }) },
    },
    plugins: [labelPlugin((d) => (d?.p && top.has(d.p.id) ? d.p.name : null))],
  });
}
async function loadPlayer(id) {
  const p = state.players.players.find((x) => x.id === id);
  try {
    const all = state.playerHistAll || (state.playerHistAll = await getJSON("player_history.json"));
    const rows = all.players[id];
    if (!rows) { toast("No game-by-game history published for this player"); return; }
    const h = { history: rows.map((r) => Object.fromEntries(all.fields.map((f, i) => [f, r[i]]))) };
    state.playerHist = [p, h];
    renderPlayerChart(p, h);
  } catch (e) { toast(e.message); }
}
function renderPlayerChart(p, h) {
  const g = h.history.slice(-10);
  $("#playerTitle").textContent = `${p.name} — last ${g.length} gameweeks`;
  $("#playerHint").textContent = `${p.team} · ${p.pos} · ${p.goals} G (${f2(p.xg, 1)} xG), ${p.assists} A (${f2(p.xa, 1)} xA) this season`;
  draw("playerChart", {
    type: "bar",
    data: { labels: g.map((x) => `GW${x.round} ${x.home ? "vs" : "@"} ${x.opponent}`), datasets: [
      { label: "xG", data: g.map((x) => x.xg), backgroundColor: css("--series-1") },
      { label: "xA", data: g.map((x) => x.xa), backgroundColor: css("--series-3") },
      { label: "Goals + assists", type: "line", data: g.map((x) => x.goals + x.assists), borderColor: css("--series-2"), backgroundColor: css("--series-2"), pointRadius: 4, borderWidth: 0, showLine: false },
    ] },
    options: { plugins: { tooltip: { callbacks: { afterTitle: (it) => `${g[it[0].dataIndex].minutes} min · ${g[it[0].dataIndex].points} FPL pts`, label: (c) => `${c.dataset.label}: ${f2(c.raw)}` } } },
      scales: { x: noGrid({ ticks: { maxRotation: 45, minRotation: 45, font: { size: 10 } } }), y: gridScale({ beginAtZero: true, title: { display: true, text: "Per match" } }) } },
  });
}

// ------------------------------------------------------------------ model lab
async function loadBacktest() {
  $("#btOut").className = "loading"; $("#btOut").textContent = "Running backtest… (takes a few seconds the first time)";
  state.backtest = L().backtest;
  if (state.backtest) renderBacktest(state.backtest);
  else $("#btOut").textContent = "Backtest not available for this league/season yet.";
}
function renderBacktest(B) {
  const w = B.blend_weights;
  $("#blendWeights").textContent = w ? "Current weights: " + Object.entries(w).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${(state.meta.models[k] || k).split(" (")[0]} ${Math.round(v * 100)}%`).join(", ") + "." : "";
  $("#btSeason").textContent = `(${B.season_label})`;
  const best = (k) => Math.min(...B.summary.map((s) => s[k]));
  const bestAcc = Math.max(...B.summary.map((s) => s.accuracy));
  $("#btOut").className = "table-wrap";
  $("#btOut").innerHTML = `<table class="data"><thead><tr><th class="l">Model</th><th>Matches</th><th>Picked winner</th><th>Log-loss</th><th>RPS</th><th>Brier</th><th>Over/Under 2.5 hit</th></tr></thead><tbody>` +
    B.summary.map((s) => `<tr><td class="l">${esc(s.label)}</td><td>${s.n}</td><td class="${s.accuracy === bestAcc ? "best" : ""}">${pct(s.accuracy, 1)}</td><td class="${s.log_loss === best("log_loss") ? "best" : ""}">${f2(s.log_loss, 3)}</td><td class="${s.rps === best("rps") ? "best" : ""}">${f2(s.rps, 3)}</td><td class="${s.brier === best("brier") ? "best" : ""}">${f2(s.brier, 3)}</td><td>${s.ou25_accuracy != null ? pct(s.ou25_accuracy, 1) : "–"}</td></tr>`).join("") +
    `</tbody></table><p class="note">“Picked winner” = how often the most likely outcome happened. Probability quality (log-loss / RPS) matters more for value: a model can pick favourites well and still be badly calibrated. Uniform guessing scores log-loss 1.099.</p>`;

  const sorted = [...B.summary].sort((a, b) => a.log_loss - b.log_loss);
  draw("btChart", { type: "bar", data: { labels: sorted.map((s) => s.label), datasets: [{ label: "Log-loss", data: sorted.map((s) => s.log_loss),
    backgroundColor: sorted.map((s) => (s.model === "market" ? css("--neutral") : css("--series-1"))) }] },
    options: { indexAxis: "y", interaction: { mode: "nearest", axis: "y", intersect: false }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `Log-loss ${f2(c.raw, 3)}${sorted[c.dataIndex].model === "market" ? " (benchmark)" : ""}` } } },
      scales: { x: gridScale({ min: Math.max(0, Math.floor(Math.min(...sorted.map((s) => s.log_loss)) * 20) / 20 - 0.05) }), y: noGrid() } } });

  const cmap = { blend: ["Blend", css("--series-1")], dc: ["Dixon-Coles", css("--series-2")], ml: ["ML Poisson", css("--series-3")], market: ["Bookmakers", css("--neutral")] };
  const ds = Object.entries(B.calibration).filter(([k]) => cmap[k]).map(([k, pts]) => ({ label: cmap[k][0], data: pts.map((p) => ({ x: p.pred, y: p.obs, n: p.n })), borderColor: cmap[k][1], backgroundColor: cmap[k][1], pointRadius: 4, showLine: true, tension: 0 }));
  ds.unshift({ label: "Perfect", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: css("--grid"), borderWidth: 1, pointRadius: 0, pointHoverRadius: 0, showLine: true });
  draw("calibChart", { type: "scatter", data: { datasets: ds }, options: { interaction: { mode: "nearest", intersect: true },
    plugins: { legend: { labels: { filter: (i) => i.text !== "Perfect" } }, tooltip: { filter: (i) => i.dataset.label !== "Perfect", callbacks: { label: (c) => `${c.dataset.label}: predicted ${pct(c.raw.x)}, happened ${pct(c.raw.y)} (${c.raw.n} games)` } } },
    scales: { x: gridScale({ min: 0, max: 1, ticks: { callback: (v) => pct(v) }, title: { display: true, text: "Predicted home-win probability" } }), y: gridScale({ min: 0, max: 1, ticks: { callback: (v) => pct(v) }, title: { display: true, text: "Actual home-win rate" } }) } } });

  for (const [id, target] of [["impHome", "home goals"], ["impAway", "away goals"]]) {
    const rows = B.ml_importance.filter((r) => r.target === target).sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));
    draw(id, { type: "bar", data: { labels: rows.map((r) => r.label), datasets: [{ data: rows.map((r) => r.coef), backgroundColor: rows.map((r) => (r.coef >= 0 ? css("--div-pos") : css("--div-neg"))) }] },
      options: { indexAxis: "y", interaction: { mode: "nearest", axis: "y", intersect: false }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.raw >= 0 ? "More" : "Fewer"} goals: ${sign(c.raw, 3)}` } } },
        scales: { x: gridScale(), y: noGrid({ ticks: { font: { size: 11 } } }) } } });
  }
}

boot().catch((e) => { $("#dataStamp").textContent = "Failed to start: " + e.message; });
