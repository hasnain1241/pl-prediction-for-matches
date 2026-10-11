renderChrome("index.html");

function probBlock(title, cls, h, d, a) {
  if (h === null || d === null || a === null) {
    return `<div class="prob ${cls}"><div class="prob-head"><span>${title}</span></div>
      <p class="small">Not available for this fixture.</p></div>`;
  }
  const label = `${title}: home ${pct(h)}, draw ${pct(d)}, away ${pct(a)}`;
  return `<div class="prob ${cls}">
    <div class="prob-head"><span>${title}</span></div>
    <div class="prob-nums">
      <span><small>H</small><b>${pct(h)}</b></span>
      <span><small>D</small><b>${pct(d)}</b></span>
      <span><small>A</small><b>${pct(a)}</b></span>
    </div>
    <div class="bar" role="img" aria-label="${label}">
      <i class="h" style="flex:${h}"></i><i class="d" style="flex:${d}"></i><i class="a" style="flex:${a}"></i>
    </div>
  </div>`;
}

// The outcome where model and market differ most, in percentage points.
function biggestGap(model, market) {
  const names = ["home win", "draw", "away win"];
  let best = null;
  model.forEach((m, i) => {
    if (m === null || market[i] === null) return;
    const diff = (m - market[i]) * 100;
    if (best === null || Math.abs(diff) > Math.abs(best.diff)) best = { diff, name: names[i] };
  });
  if (!best) return "";
  const dir = best.diff > 0 ? "higher" : "lower";
  return `<p class="gap">Biggest gap: model is <strong>${Math.abs(best.diff).toFixed(0)} points ${dir}</strong>
    than the market on the ${best.name}.</p>`;
}

function card(r) {
  const model = [num(r["P(H)"]), num(r["P(D)"]), num(r["P(A)"])];
  const market = [num(r["market_P(H)"]), num(r["market_P(D)"]), num(r["market_P(A)"])];
  return `<article class="card">
    <div class="teams">
      <span class="team home">${esc(r.HomeTeam)}<span class="side">Home</span></span>
      <span class="vs">v</span>
      <span class="team away">${esc(r.AwayTeam)}<span class="side">Away</span></span>
    </div>
    ${probBlock("Model", "model", ...model)}
    ${probBlock("Market (bookmaker)", "market", ...market)}
    ${biggestGap(model, market)}
    <dl class="stats">
      <div><dt>Expected goals</dt><dd>${fmt(num(r.xG_home), 2)} – ${fmt(num(r.xG_away), 2)}</dd></div>
      <div><dt>Likeliest score</dt><dd>${esc(r.predicted_scoreline || "–")}</dd></div>
      <div><dt>Over 2.5 goals</dt><dd>${pct(num(r["P(over 2.5)"]))}</dd></div>
      <div><dt>Both teams score</dt><dd>${pct(num(r["P(BTTS)"]))}</dd></div>
    </dl>
  </article>`;
}

loadCSV("predictions_next_gameweek.csv").then(rows => {
  const el = document.getElementById("fixtures");
  if (!rows.length) { el.innerHTML = "<p>No fixtures in the predictions file.</p>"; return; }
  const days = new Map();
  rows.forEach(r => { if (!days.has(r.Date)) days.set(r.Date, []); days.get(r.Date).push(r); });
  el.innerHTML = [...days.keys()].sort().map(date =>
    `<h2 class="day">${esc(formatDate(date))}</h2><div class="fixtures">${days.get(date).map(card).join("")}</div>`
  ).join("");
  return new Set(rows.map(r => r.HomeTeam + "|" + r.AwayTeam));
}).catch(err => { showError(document.getElementById("fixtures"), err); return new Set(); })
  .then(predicted => loadCSV("fixtures_remaining.csv").then(rows => renderSchedule(rows, predicted || new Set())))
  .catch(err => showError(document.getElementById("schedule"), err));

// Remaining fixtures that have no prediction yet, grouped by gameweek.
function renderSchedule(rows, predicted) {
  const el = document.getElementById("schedule");
  const rest = rows.filter(r => !predicted.has(r.HomeTeam + "|" + r.AwayTeam));
  if (!rest.length) { el.innerHTML = "<p>No further fixtures listed.</p>"; return; }
  const weeks = new Map();
  rest.forEach(r => { const k = r.gameweek || "Date to be confirmed"; if (!weeks.has(k)) weeks.set(k, []); weeks.get(k).push(r); });
  const short = { weekday: "short", day: "numeric", month: "short" };
  el.innerHTML = [...weeks.entries()].map(([week, list], i) => {
    const dates = list.map(r => r.Date).filter(Boolean).sort();
    const span = dates.length ? formatDate(dates[0], short) + (dates[dates.length - 1] !== dates[0] ? " to " + formatDate(dates[dates.length - 1], short) : "") : "";
    const title = /^\d+$/.test(week) ? `Gameweek ${week}` : week;
    return `<details class="week"${i === 0 ? " open" : ""}>
      <summary><span>${esc(title)}</span><span class="small">${esc(span)} · ${list.length} ${list.length === 1 ? "match" : "matches"}</span></summary>
      <ul>${list.map(r => `<li>
        <span class="when">${r.Date ? esc(formatDate(r.Date, short)) : "TBC"} <span class="num">${esc(r.kickoff_uk)}</span></span>
        <span class="team home">${esc(r.HomeTeam)}</span><span class="vs">v</span><span class="team away">${esc(r.AwayTeam)}</span>
      </li>`).join("")}</ul>
    </details>`;
  }).join("");
}
