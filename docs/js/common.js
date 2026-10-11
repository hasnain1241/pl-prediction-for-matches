// Shared helpers: CSV loading, number formatting, header and footer.
// All data paths are relative so the site works under /<repo-name>/ on GitHub Pages.

const DATA_DIR = "data/";

function parseCSV(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift() || [];
  return rows.map(r => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

async function loadCSV(name) {
  const res = await fetch(DATA_DIR + name, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  return parseCSV(await res.text());
}

async function loadMeta() {
  try {
    const res = await fetch(DATA_DIR + "meta.json", { cache: "no-cache" });
    return res.ok ? await res.json() : {};
  } catch { return {}; }
}

// Blank cells (hard predictions have no log loss) become null, never 0.
function num(v) {
  if (v === undefined || v === null || String(v).trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

const fmt = (v, d = 4) => (v === null ? "–" : v.toFixed(d));
const pct = (v, d = 0) => (v === null ? "–" : (v * 100).toFixed(d) + "%");

// Parse YYYY-MM-DD as a local date so it never shifts a day across time zones.
function parseDate(s) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s || "");
  return m ? new Date(+m[1], +m[2] - 1, +m[3]) : null;
}

function formatDate(s, opts = { weekday: "long", day: "numeric", month: "long", year: "numeric" }) {
  const d = parseDate(s);
  return d ? d.toLocaleDateString("en-GB", opts) : s;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const FEATURE_SETS = {
  A_form_only: { label: "A · form only", color: "var(--set-a)" },
  B_form_plus_elo: { label: "B · form + Elo", color: "var(--set-b)" },
  C_with_market: { label: "C · with market odds", color: "var(--set-c)" },
  "-": { label: "Baseline", color: "var(--muted)" },
};

function setLabel(key) {
  return FEATURE_SETS[key] ? FEATURE_SETS[key].label : key;
}

function showError(el, err) {
  el.innerHTML = `<div class="error"><strong>Could not load the data.</strong> ${esc(err.message)}.
    If you opened this file directly, serve the folder instead: <code>python3 -m http.server -d docs</code>.</div>`;
}

function renderChrome(current) {
  const pages = [["index.html", "Predictions"], ["performance.html", "Model performance"], ["about.html", "About"]];
  document.getElementById("site-header").innerHTML = `<div class="wrap">
    <a class="brand" href="index.html">Premier League match predictions</a>
    <nav aria-label="Main">${pages.map(([href, name]) =>
      `<a href="${href}"${href === current ? ' aria-current="page"' : ""}>${name}</a>`).join("")}</nav>
  </div>`;
  document.getElementById("site-footer").innerHTML = `<div class="wrap">
    <p><strong>A statistics project, not betting advice.</strong> No model here beat the bookmaker's closing odds.</p>
    <p id="trained-through"></p>
    <p>Data: football-data.co.uk, Understat, and the vaastav Fantasy-Premier-League archive.</p>
  </div>`;
  loadMeta().then(meta => {
    if (!meta.trained_through) return;
    const text = `Based on results up to ${formatDate(meta.trained_through, { day: "numeric", month: "long", year: "numeric" })}.`;
    document.querySelectorAll("#trained-through, [data-trained-through]").forEach(el => { el.textContent = text; });
  });
}
