/* Poisson / Dixon-Coles market maths. Mirrors backend/models.py (tests/ checks they agree).
   Works as a plain <script> in the browser and as a module in Node for the tests. */
(function (root) {
const MAXG = 10;
function poissonPmf(lam) {
  const out = [Math.exp(-lam)];
  for (let k = 1; k <= MAXG; k++) out.push(out[k - 1] * lam / k);
  return out;
}
function americanOdds(p) {
  if (!(p > 0 && p < 1)) return null;
  return p >= 0.5 ? Math.round(-100 * p / (1 - p)) : Math.round(100 * (1 - p) / p);
}
const price = (p) => ({ p, decimal: p > 0 ? 1 / p : null, american: americanOdds(p) });
function markets(lh, la, rho = 0) {
  const ph = poissonPmf(lh), pa = poissonPmf(la);
  const m = ph.map((x) => pa.map((y) => x * y));
  if (rho) {
    m[0][0] *= 1 - lh * la * rho; m[0][1] *= 1 + lh * rho; m[1][0] *= 1 + la * rho; m[1][1] *= 1 - rho;
  }
  let tot = 0; m.forEach((r) => r.forEach((v, j) => { r[j] = Math.max(0, v); tot += r[j]; }));
  m.forEach((r) => r.forEach((v, j) => (r[j] = v / tot)));
  const sum = (f) => { let t = 0; m.forEach((r, h) => r.forEach((v, a) => { if (f(h, a)) t += v; })); return t; };
  const lines = ["0.5", "1.5", "2.5", "3.5", "4.5"];
  const flat = [];
  for (let h = 0; h < 7; h++) for (let a = 0; a < 7; a++) flat.push([m[h][a], `${h}-${a}`]);
  // most likely first; exact ties broken by score label so Python and JS always agree
  const r12 = (v) => Math.round(v * 1e12);
  flat.sort((x, y) => r12(y[0]) - r12(x[0]) || (x[1] < y[1] ? -1 : 1));
  return {
    lambda_home: lh, lambda_away: la, rho,
    home: price(sum((h, a) => h > a)), draw: price(sum((h, a) => h === a)), away: price(sum((h, a) => h < a)),
    btts_yes: price(sum((h, a) => h > 0 && a > 0)),
    home_cs: price(sum((h, a) => a === 0)), away_cs: price(sum((h, a) => h === 0)),
    over: Object.fromEntries(lines.map((t) => [t, price(sum((h, a) => h + a > +t))])),
    under: Object.fromEntries(lines.map((t) => [t, price(sum((h, a) => h + a < +t))])),
    home_goals_dist: m.slice(0, 7).map((r) => r.reduce((x, y) => x + y, 0)),
    away_goals_dist: [0, 1, 2, 3, 4, 5, 6].map((a) => m.reduce((x, r) => x + r[a], 0)),
    matrix: m.slice(0, 7).map((r) => r.slice(0, 7)),
    top_scores: flat.slice(0, 8).map(([p, score]) => ({ score, ...price(p) })),
  };
}
function implied(o) {
  if (!o || o.some((x) => !(x > 0))) return null;
  const inv = o.map((x) => 1 / x), t = inv.reduce((a, b) => a + b, 0);
  return inv.map((x) => x / t);
}
function rolling(arr, w) {
  return arr.map((_, i) => {
    const sl = arr.slice(Math.max(0, i - w + 1), i + 1).filter((x) => x != null);
    return sl.length ? sl.reduce((a, b) => a + b, 0) / sl.length : null;
  });
}
  const api = { poissonPmf, americanOdds, price, markets, implied, rolling };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== "undefined" ? globalThis : this);
