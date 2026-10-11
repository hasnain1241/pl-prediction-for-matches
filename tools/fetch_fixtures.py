"""Write docs/data/fixtures_remaining.csv: every Premier League fixture of the season not yet played.

The fixture list comes from the public Fantasy Premier League API. It is a schedule only. The
prediction pipeline is not involved and no probabilities are produced here.

Usage (from the repository root, standard library only):
    python3 tools/fetch_fixtures.py
"""
import csv
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

API = "https://fantasy.premierleague.com/api/"
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "fixtures_remaining.csv"
UK = ZoneInfo("Europe/London")

# FPL club names that differ from the football-data.co.uk names used in the prediction files.
NAME_MAP = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Norwich City": "Norwich",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Cardiff City": "Cardiff",
    "Sheffield Utd": "Sheffield United",
}


def get(path: str):
    request = urllib.request.Request(API + path, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    teams = {t["id"]: NAME_MAP.get(t["name"], t["name"]) for t in get("bootstrap-static/")["teams"]}
    rows = []
    for f in get("fixtures/"):
        if f.get("finished"):
            continue
        kickoff = f.get("kickoff_time")
        local = datetime.fromisoformat(kickoff.replace("Z", "+00:00")).astimezone(UK) if kickoff else None
        rows.append({
            "gameweek": f.get("event") or "",
            "Date": local.date().isoformat() if local else "",
            "kickoff_uk": local.strftime("%H:%M") if local else "",
            "HomeTeam": teams[f["team_h"]],
            "AwayTeam": teams[f["team_a"]],
        })
    # Fixtures without a date yet (postponed, to be rearranged) go last.
    rows.sort(key=lambda r: (r["Date"] == "", r["Date"], r["kickoff_uk"], r["HomeTeam"]))
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gameweek", "Date", "kickoff_uk", "HomeTeam", "AwayTeam"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
