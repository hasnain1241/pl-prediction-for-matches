renderChrome("index.html");

const PILL = { "Home win": "h", "Draw": "d", "Away win": "a" };
const fixturesEl = document.getElementById("fixtures");
const gwSelect = document.getElementById("gw");
const teamSelect = document.getElementById("team");
let rows = [];

function fixtureRow(r) {
  const result = r.predicted_result;
  const pickHome = result === "Home win", pickAway = result === "Away win";
  const when = r.Date ? formatDate(r.Date, { weekday: "short", day: "numeric", month: "short" }) : "Date to be confirmed";
  const odds = /market/.test(r.based_on);
  const verdict = result === "Draw" ? "Draw" : `${esc(r.predicted_winner)} win`;
  return `<article class="fx">
    <div class="when">${esc(when)}</div>
    <div class="team home${pickHome ? " pick" : ""}">${esc(r.HomeTeam)}</div>
    <div class="score" aria-label="Predicted score ${esc(r.predicted_score)}">${esc(r.predicted_score)}</div>
    <div class="team away${pickAway ? " pick" : ""}">${esc(r.AwayTeam)}</div>
    <div class="verdict"><span class="pill ${PILL[result] || ""}">${verdict}</span>
      <span class="basis">${odds ? "Uses bookmaker odds" : "Form and Elo only"}</span></div>
  </article>`;
}

function render() {
  const gw = gwSelect.value, team = teamSelect.value;
  const list = rows.filter(r => (gw === "all" || r.gameweek === gw) && (team === "all" || r.HomeTeam === team || r.AwayTeam === team));
  if (!list.length) { fixturesEl.innerHTML = "<p>No fixtures match this filter.</p>"; return; }
  const weeks = new Map();
  list.forEach(r => { const k = r.gameweek || "TBC"; if (!weeks.has(k)) weeks.set(k, []); weeks.get(k).push(r); });
  fixturesEl.innerHTML = [...weeks.entries()].map(([week, items]) => {
    const dates = items.map(r => r.Date).filter(Boolean).sort();
    const short = { day: "numeric", month: "short" };
    const span = dates.length ? formatDate(dates[0], short) + (dates.at(-1) !== dates[0] ? " to " + formatDate(dates.at(-1), short) : "") : "";
    return `<div class="gw"><h2>${/^\d+$/.test(week) ? "Gameweek " + week : "Date to be confirmed"}</h2><span class="small">${esc(span)}</span></div>
      <div class="fixtures">${items.map(fixtureRow).join("")}</div>`;
  }).join("");
}

function summarise() {
  const count = label => rows.filter(r => r.predicted_result === label).length;
  const tiles = [
    ["Matches predicted", rows.length, "Every unplayed match this season"],
    ["Home wins", count("Home win"), "Predicted"],
    ["Away wins", count("Away win"), "Predicted"],
    ["Draws", count("Draw"), "A draw is rarely the single likeliest result"],
  ];
  document.getElementById("summary").innerHTML = tiles.map(([label, value, sub]) =>
    `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${esc(sub)}</div></div>`).join("");
}

loadCSV("predictions_remaining_season.csv").then(data => {
  rows = data;
  if (!rows.length) { fixturesEl.innerHTML = "<p>No fixtures in the predictions file.</p>"; return; }
  const weeks = [...new Set(rows.map(r => r.gameweek).filter(Boolean))].sort((a, b) => a - b);
  const teams = [...new Set(rows.flatMap(r => [r.HomeTeam, r.AwayTeam]))].sort();
  gwSelect.innerHTML = '<option value="all">All gameweeks</option>' + weeks.map(w => `<option value="${w}">Gameweek ${w}</option>`).join("");
  teamSelect.innerHTML = '<option value="all">All teams</option>' + teams.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join("");
  gwSelect.value = weeks[0];                     // start on the next gameweek
  gwSelect.addEventListener("change", render);
  teamSelect.addEventListener("change", () => { if (teamSelect.value !== "all" && gwSelect.value !== "all") gwSelect.value = "all"; render(); });
  summarise();
  render();
}).catch(err => showError(fixturesEl, err));
