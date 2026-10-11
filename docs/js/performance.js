renderChrome("performance.html");

const isBookmaker = r => /^bookmaker/i.test(r.model);
const isBaseline = r => r.feature_set === "-";

// ---------- sortable model table ----------

const COLUMNS = [
  { key: "model", label: "Model", text: true },
  { key: "feature_set", label: "Feature set", text: true },
  { key: "cv_logloss", label: "CV log loss" },
  { key: "test_logloss", label: "Holdout log loss" },
  { key: "test_rps", label: "Holdout RPS" },
  { key: "test_brier", label: "Holdout Brier" },
  { key: "test_acc", label: "Holdout accuracy", pct: true, desc: true },
  { key: "test_macro_f1", label: "Holdout macro F1", desc: true, digits: 3 },
  { key: "test_rec_draw", label: "Draw recall", pct: true, desc: true },
  { key: "live_logloss", label: "Live log loss" },
  { key: "note", label: "Note", text: true, noSort: true },
];

const state = { rows: [], sort: "cv_logloss", dir: 1, filter: "all" };

function cell(r, c) {
  if (c.key === "model") {
    return `<td>${esc(r.model)}${isBookmaker(r) ? '<span class="tag">Benchmark</span>' : ""}</td>`;
  }
  if (c.key === "feature_set") {
    const fs = FEATURE_SETS[r.feature_set];
    return `<td><span class="set"><i class="swatch dot" style="background:${fs ? fs.color : "var(--muted)"}"></i>${esc(setLabel(r.feature_set))}</span></td>`;
  }
  if (c.key === "note") return `<td class="note">${esc(r.note)}</td>`;
  const v = num(r[c.key]);
  if (v === null) return '<td class="n blank" title="Not defined for hard predictions">–</td>';
  return `<td class="n">${c.pct ? pct(v, 1) : fmt(v, c.digits ?? 4)}</td>`;
}

function renderModels() {
  const col = COLUMNS.find(c => c.key === state.sort);
  const rows = state.rows.filter(r => isBookmaker(r) || state.filter === "all" || r.feature_set === state.filter);
  rows.sort((a, b) => {
    if (col.text) return state.dir * String(a[col.key]).localeCompare(String(b[col.key]));
    const x = num(a[col.key]), y = num(b[col.key]);
    if (x === null || y === null) return (x === null) - (y === null); // blanks always last
    return state.dir * (x - y);
  });
  const head = COLUMNS.map(c => {
    const active = c.key === state.sort;
    const aria = active ? ` aria-sort="${state.dir === 1 ? "ascending" : "descending"}"` : "";
    const cls = c.text ? "" : ' class="n"';
    if (c.noSort) return `<th${cls}>${c.label}</th>`;
    return `<th${cls}${aria}><button type="button" data-key="${c.key}">${c.label}<span class="arrow">${active ? (state.dir === 1 ? "▲" : "▼") : ""}</span></button></th>`;
  }).join("");
  document.getElementById("models").innerHTML = `<thead><tr>${head}</tr></thead><tbody>${rows.map(r =>
    `<tr${isBookmaker(r) ? ' class="benchmark"' : ""}>${COLUMNS.map(c => cell(r, c)).join("")}</tr>`).join("")}</tbody>`;
}

function setupModels(rows) {
  state.rows = rows;
  const sets = [...new Set(rows.map(r => r.feature_set))].sort((a, b) => (a === "-") - (b === "-") || a.localeCompare(b));
  const filter = document.getElementById("filter");
  filter.innerHTML = '<option value="all">All</option>' +
    sets.map(s => `<option value="${esc(s)}">${esc(s === "-" ? "Baselines" : setLabel(s))}</option>`).join("");
  filter.addEventListener("change", () => { state.filter = filter.value; renderModels(); });
  document.getElementById("models").addEventListener("click", e => {
    const btn = e.target.closest("button[data-key]");
    if (!btn) return;
    const col = COLUMNS.find(c => c.key === btn.dataset.key);
    if (state.sort === col.key) state.dir = -state.dir;
    else { state.sort = col.key; state.dir = col.desc ? -1 : 1; }
    renderModels();
  });
  renderModels();
}

// ---------- headline tiles ----------

function renderTiles(rows) {
  const bookie = rows.find(isBookmaker);
  const models = rows.filter(r => !isBaseline(r) && num(r.test_logloss) !== null);
  const best = models.reduce((a, b) => (num(b.test_logloss) < num(a.test_logloss) ? b : a));
  const accs = models.map(r => num(r.test_acc)).filter(v => v !== null);
  const drawRecalls = models.map(r => num(r.test_rec_draw)).filter(v => v !== null);
  const tiles = [
    ["Bookmaker holdout log loss", fmt(num(bookie.test_logloss)), "Closing odds, the benchmark"],
    ["Best model holdout log loss", fmt(num(best.test_logloss)), `${best.model}, ${setLabel(best.feature_set)}`],
    ["Model holdout accuracy", `${pct(Math.min(...accs), 1)}–${pct(Math.max(...accs), 1)}`, `Bookmaker: ${pct(num(bookie.test_acc), 1)}`],
    ["Highest draw recall", pct(Math.max(...drawRecalls), 1), "Share of actual draws any model picked"],
  ];
  document.getElementById("tiles").innerHTML = tiles.map(([label, value, sub]) =>
    `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${esc(sub)}</div></div>`).join("");
}

// ---------- dot plot: holdout log loss by model ----------

function renderChart(rows) {
  const bookie = num(rows.find(isBookmaker).test_logloss);
  const pts = rows.filter(r => !isBaseline(r) && num(r.test_logloss) !== null)
    .map(r => ({ model: r.model, set: r.feature_set, v: num(r.test_logloss) }));
  const best = m => Math.min(...pts.filter(p => p.model === m).map(p => p.v));
  const models = [...new Set(pts.map(p => p.model))].sort((a, b) => best(a) - best(b));
  const sets = [...new Set(pts.map(p => p.set))].sort();

  document.getElementById("chart-legend").innerHTML = sets.map(s =>
    `<span><i class="swatch dot" style="background:${FEATURE_SETS[s] ? FEATURE_SETS[s].color : "var(--muted)"}"></i>${esc(setLabel(s))}</span>`
  ).join("") + '<span><i class="swatch rule"></i>Bookmaker (closing odds)</span>';

  const left = rows.filter(r => isBaseline(r) && !isBookmaker(r) && num(r.test_logloss) !== null)
    .map(r => `${r.model} (${fmt(num(r.test_logloss))})`);
  document.getElementById("chart-note").textContent = left.length
    ? `Not plotted, to keep the scale readable: ${left.join(", ")}. Models without probabilities have no log loss.` : "";

  const box = document.getElementById("chart");
  const tip = document.getElementById("tooltip");
  const step = 0.005;
  const lo = Math.floor((Math.min(bookie, ...pts.map(p => p.v)) - 0.001) / step) * step;
  const hi = Math.ceil((Math.max(...pts.map(p => p.v)) + 0.001) / step) * step;

  function draw() {
    const W = Math.max(box.clientWidth, 280);
    const padL = 10, padR = 16, top = 26, rowH = 44, bottom = 34;
    const H = top + models.length * rowH + bottom;
    const x = v => padL + ((v - lo) / (hi - lo)) * (W - padL - padR);
    const tickEvery = W < 520 ? 2 : 1;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Holdout log loss by model and feature set, with the bookmaker at ${fmt(bookie)}">`;
    for (let i = 0, t = lo; t <= hi + 1e-9; i++, t = lo + i * step) {
      s += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - bottom}" stroke="var(--line)"/>`;
      if (i % tickEvery === 0) s += `<text x="${x(t)}" y="${H - bottom + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">${t.toFixed(3)}</text>`;
    }
    s += `<text x="${W - padR}" y="${H - 2}" text-anchor="end" font-size="11" fill="var(--muted)">Holdout log loss (lower is better) →</text>`;
    s += `<line x1="${x(bookie)}" x2="${x(bookie)}" y1="${top - 6}" y2="${H - bottom}" stroke="var(--bench-line)" stroke-width="2"/>`;
    s += `<text x="${x(bookie) + 6}" y="${top - 10}" font-size="12" font-weight="600" fill="var(--text)">Bookmaker ${fmt(bookie)}</text>`;
    models.forEach((m, i) => {
      const y = top + i * rowH;
      s += `<text x="${padL}" y="${y + 16}" font-size="12.5" fill="var(--text)" paint-order="stroke" stroke="var(--surface)" stroke-width="5" stroke-linejoin="round">${esc(m)}</text>`;
      s += `<line x1="${padL}" x2="${W - padR}" y1="${y + 30}" y2="${y + 30}" stroke="var(--axis)" stroke-width="1"/>`;
      pts.filter(p => p.model === m).forEach(p => {
        const color = FEATURE_SETS[p.set] ? FEATURE_SETS[p.set].color : "var(--muted)";
        s += `<circle class="dot" tabindex="0" cx="${x(p.v)}" cy="${y + 30}" r="6" fill="${color}" stroke="var(--surface)" stroke-width="2"
          data-tip="${esc(m)}|${esc(setLabel(p.set))}|${fmt(p.v)}|${(p.v - bookie >= 0 ? "+" : "") + (p.v - bookie).toFixed(4)}"/>`;
        s += `<circle cx="${x(p.v)}" cy="${y + 30}" r="14" fill="transparent" data-for="1"/>`;
      });
    });
    box.innerHTML = s + "</svg>";
  }

  function show(dot) {
    const [m, set, v, d] = dot.dataset.tip.split("|");
    tip.innerHTML = `<strong>${m}</strong><br>${set}<br>Log loss ${v} (${d} vs bookmaker)`;
    tip.style.display = "block";
    const panel = tip.parentElement.getBoundingClientRect(), r = dot.getBoundingClientRect();
    const leftPx = Math.min(Math.max(r.left - panel.left - tip.offsetWidth / 2 + 6, 4), panel.width - tip.offsetWidth - 4);
    tip.style.left = leftPx + "px";
    tip.style.top = (r.top - panel.top - tip.offsetHeight - 8) + "px";
  }
  const dotOf = t => (t.dataset && t.dataset.for ? t.previousElementSibling : t.closest && t.closest(".dot"));
  box.addEventListener("pointerover", e => { const d = dotOf(e.target); if (d) show(d); });
  box.addEventListener("pointerout", () => { tip.style.display = "none"; });
  box.addEventListener("focusin", e => { const d = dotOf(e.target); if (d) show(d); });
  box.addEventListener("focusout", () => { tip.style.display = "none"; });

  draw();
  let timer;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(draw, 100); });
}

// ---------- plain tables ----------

function plainTable(id, rows, columns) {
  const body = rows.map(r => `<tr>${columns.map(c => {
    if (c.text) return `<td>${esc(c.key === "feature_set" ? setLabel(r[c.key]) : r[c.key])}</td>`;
    const v = num(r[c.key]);
    if (v === null) return '<td class="n blank">–</td>';
    return `<td class="n">${c.pct ? pct(v, 1) : c.int ? v.toFixed(0) : fmt(v, c.digits ?? 4)}</td>`;
  }).join("")}</tr>`).join("");
  document.getElementById(id).innerHTML =
    `<thead><tr>${columns.map(c => `<th${c.text ? "" : ' class="n"'}>${c.label}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
}

const main = document.querySelector("main");

loadCSV("model_comparison.csv").then(rows => {
  renderTiles(rows);
  renderChart(rows);
  setupModels(rows);
}).catch(err => showError(document.getElementById("tiles"), err));

loadCSV("goals_regression_comparison.csv").then(rows => plainTable("goals", rows, [
  { key: "model", label: "Model", text: true },
  { key: "feature_set", label: "Feature set", text: true },
  { key: "test_gd_mae", label: "GD MAE", digits: 3 },
  { key: "test_gd_rmse", label: "GD RMSE", digits: 3 },
  { key: "test_gd_sign_acc", label: "Sign accuracy", pct: true },
  { key: "test_gd_sign_acc_decisive", label: "Sign accuracy, no draws", pct: true },
  { key: "test_mae_home", label: "MAE home goals", digits: 3 },
  { key: "test_mae_away", label: "MAE away goals", digits: 3 },
  { key: "test_poisson_dev_home", label: "Poisson deviance home", digits: 3 },
  { key: "test_poisson_dev_away", label: "Poisson deviance away", digits: 3 },
])).catch(err => showError(document.getElementById("goals").parentElement, err));

loadCSV("xg_experiment.csv").then(rows => plainTable("xg", rows, [
  { key: "model", label: "Model", text: true },
  { key: "feature_set", label: "Feature set", text: true },
  { key: "n_features", label: "Features", int: true },
  { key: "cv_logloss", label: "CV log loss" },
  { key: "test_logloss", label: "Holdout log loss" },
  { key: "test_rps", label: "Holdout RPS" },
  { key: "test_brier", label: "Holdout Brier" },
  { key: "test_acc", label: "Holdout accuracy", pct: true },
])).catch(err => showError(document.getElementById("xg").parentElement, err));
