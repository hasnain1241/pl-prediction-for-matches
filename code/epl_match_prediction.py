# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Premier League Match Prediction
#
# An end-to-end pipeline that predicts English Premier League results from historical data.
#
# **Before running:** open **Settings > Internet** in the Kaggle sidebar and switch it **on** so the
# notebook can download the season files. If Internet is off, the notebook falls back to cached files
# in `/kaggle/working/data/raw` and then to any CSVs attached under `/kaggle/input/`.
#
# **Targets**
#
# 1. Primary: three-class classification of the full-time result (Home / Draw / Away) with calibrated probabilities.
# 2. Secondary: goals regression for home and away goals (Poisson GLM and XGBoost `count:poisson`), converted to
#    H/D/A probabilities through a Dixon-Coles style scoreline grid.
# 3. Baseline: plain linear regression on goal difference, scored by sign accuracy.
#
# **Contents**
#
# 0. Setup, configuration and reproducibility
# 1. Data loading, schema harmonisation and optional sources
# 2. Cleaning and exploratory analysis
# 3. Leakage-safe feature engineering and leakage tests
# 4. Validation design
# 5. Classification models
# 6. Goals regression models
# 7. Evaluation
# 8. Predicting upcoming fixtures (next gameweek, then a plain result for every remaining match)
# 9. Results and limitations

# %% [markdown]
# ## 0. Setup, configuration and reproducibility
#
# All tunable settings live in one `CONFIG` dict. Missing packages are installed quietly. Optional packages
# (LightGBM, `understatapi`) are only used if they import successfully.

# %%
import importlib
import subprocess
import sys


def ensure_packages(required: dict, optional: dict) -> None:
    """Install any missing package with `pip install -q`. Optional packages may fail silently."""
    for group, must in ((required, True), (optional, False)):
        for module, pip_name in group.items():
            try:
                importlib.import_module(module)
            except Exception:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name],
                                   check=must, timeout=240, capture_output=True)
                except Exception as exc:  # offline or blocked
                    if must:
                        print(f"Could not install required package {pip_name}: {exc}")


ensure_packages(
    required={"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn", "xgboost": "xgboost",
              "optuna": "optuna", "shap": "shap", "matplotlib": "matplotlib", "seaborn": "seaborn",
              "scipy": "scipy", "joblib": "joblib", "requests": "requests", "pyarrow": "pyarrow"},
    optional={"understatapi": "understatapi"},
)

# %%
import csv
import glob
import io
import json
import os
import random
import time
import traceback
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
import seaborn as sns
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from requests.adapters import HTTPAdapter
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_poisson_deviance, mean_squared_error, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from urllib3.util.retry import Retry

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

ON_KAGGLE = os.path.exists("/kaggle")
BASE_DIR = Path("/kaggle/working") if ON_KAGGLE else Path("working")
_today = date.today()

CONFIG = {
    # data
    "first_season_start": 2000,
    "last_season_start": max(2026, _today.year if _today.month >= 8 else _today.year - 1),
    "full_season_rows": 380,
    "raw_dir": BASE_DIR / "data" / "raw",
    "out_dir": BASE_DIR,
    "input_dir": Path("/kaggle/input"),
    "user_agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
    "use_clubelo": True,
    "use_understat": True,
    "use_fpl": True,
    "xg_first_season": 2014,
    # features
    "form_windows": [3, 5, 10],
    "venue_windows": [5],
    "ewm_halflife": 5,
    "h2h_matches": 5,
    "strength_prior_matches": 6,
    "elo_k": 20.0,
    "elo_home_adv": 70.0,
    "elo_init": 1500.0,
    "rest_cap_days": 21,
    "congestion_days": 14,
    # validation
    "burn_in_seasons": 1,          # first season only feeds the rolling features
    "n_test_seasons": 2,           # most recent two completed seasons
    "cv_first_val_season": 2016,   # train through 2015/16, validate 2016/17, and so on
    "cv_warmup_folds": 2,          # first folds only warm up the calibrators and are not scored
    # models
    "seed": 42,
    "n_jobs": os.cpu_count() or 4,
    "logreg_C_grid": [0.003, 0.01, 0.03],
    "poisson_alpha_grid": [0.03, 0.1, 0.3],
    "optuna_trials": 40,
    "optuna_timeout_s": 90,        # per feature set
    "optuna_folds": 3,             # most recent walk-forward folds used inside the Optuna objective
    "xgb_max_rounds": 800,
    "xgb_early_stopping": 30,
    "max_goals": 8,
    # evaluation
    "n_bootstrap": 1000,
    "shap_sample": 2000,
    "value_bet_threshold": 1.05,
    "fast_mode": os.environ.get("EPL_FAST", "0") == "1",   # developer switch for quick smoke runs
}
if CONFIG["fast_mode"]:
    CONFIG.update({"optuna_trials": 4, "optuna_timeout_s": 20, "n_bootstrap": 200, "xgb_max_rounds": 300})

for _dir in (CONFIG["raw_dir"], CONFIG["out_dir"]):
    Path(_dir).mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    """Seed every random number generator used in the notebook."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seeds(CONFIG["seed"])
RNG = np.random.default_rng(CONFIG["seed"])
NOTEBOOK_T0 = time.time()

# Chart style: one palette used by every figure (categorical slots are assigned in a fixed order).
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
           "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
SERIES = list(PALETTE.values())
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
OUTCOME_COLORS = {"Home": PALETTE["blue"], "Draw": PALETTE["orange"], "Away": PALETTE["aqua"]}
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
    "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "figure.dpi": 100,
    "font.size": 10,
})

CLASS_NAMES = ["Away", "Draw", "Home"]   # y = 0, 1, 2 (an ordered scale, which RPS relies on)
AVAILABILITY_LOG: List[dict] = []


def log_source(source: str, status: str, detail: str) -> None:
    """Record whether a data source loaded, for the Data availability log."""
    AVAILABILITY_LOG.append({"source": source, "status": status, "detail": detail})
    print(f"[{source}] {status}: {detail}")


def show(df: pd.DataFrame, title: Optional[str] = None) -> None:
    """Display a DataFrame in a notebook, or print it when run as a script."""
    if title:
        print(f"\n{title}")
    try:
        get_ipython()  # noqa: F821
        from IPython.display import display
        display(df)
    except NameError:
        print(df.to_string())


print(f"Kaggle: {ON_KAGGLE} | output dir: {CONFIG['out_dir']} | seasons: "
      f"{CONFIG['first_season_start']}/{(CONFIG['first_season_start'] + 1) % 100:02d} to "
      f"{CONFIG['last_season_start']}/{(CONFIG['last_season_start'] + 1) % 100:02d} | LightGBM: {HAS_LGBM}")

# %% [markdown]
# ### Reproducibility
#
# - A single seed (`CONFIG["seed"]`) is applied to Python, NumPy, scikit-learn, XGBoost and Optuna.
# - Every split is chronological. Nothing is shuffled, so the folds are identical on every run.
# - Raw downloads are cached under `data/raw`. The in-progress season file is refreshed on each run, so the
#   "live" block and the next-gameweek predictions change week to week while the historical results stay fixed.
# - Optuna is time-boxed. On a slower machine it may finish fewer trials, which can change the tuned
#   XGBoost parameters slightly. Raise `optuna_timeout_s` for a fully trial-count-driven search.

# %% [markdown]
# ## 1. Data loading
#
# The core source is football-data.co.uk, which publishes one CSV per season. Column availability changes over
# time (closing odds only exist from 2019/20, market averages change name in 2019/20), so the loader keeps the
# union of the columns the pipeline needs, fills the gaps with NaN and logs what each season lacks. Team names
# from every source pass through one mapping dict.

# %%
# Canonical names follow football-data.co.uk. Every other spelling maps onto them.
TEAM_NAME_MAP: Dict[str, str] = {
    "Manchester United": "Man United", "Manchester Utd": "Man United", "Man Utd": "Man United",
    "ManUnited": "Man United", "Manchester City": "Man City", "ManCity": "Man City",
    "Tottenham Hotspur": "Tottenham", "Spurs": "Tottenham",
    "Newcastle United": "Newcastle", "Newcastle Utd": "Newcastle",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "Nottingham Forest": "Nott'm Forest", "Nott'ham Forest": "Nott'm Forest", "Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom", "West Bromwich": "West Brom", "WestBrom": "West Brom",
    "West Ham United": "West Ham", "WestHam": "West Ham",
    "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Leicester City": "Leicester", "Leeds United": "Leeds", "Norwich City": "Norwich",
    "Sheffield Utd": "Sheffield United", "SheffieldUnited": "Sheffield United", "Sheff Utd": "Sheffield United",
    "Queens Park Rangers": "QPR", "Stoke City": "Stoke", "Swansea City": "Swansea",
    "Cardiff City": "Cardiff", "Hull City": "Hull", "Huddersfield Town": "Huddersfield",
    "AFC Bournemouth": "Bournemouth", "Luton Town": "Luton", "Ipswich Town": "Ipswich",
    "Birmingham City": "Birmingham", "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton",
    "Wigan Athletic": "Wigan", "Derby County": "Derby", "Charlton Athletic": "Charlton",
    "Coventry City": "Coventry", "Bradford City": "Bradford", "Sunderland AFC": "Sunderland",
    "CrystalPalace": "Crystal Palace", "AstonVilla": "Aston Villa", "Middlesboro": "Middlesbrough",
}
# ClubElo uses its own URL names for some clubs (canonical name -> ClubElo name).
CLUBELO_NAMES: Dict[str, str] = {
    "Man United": "ManUnited", "Man City": "ManCity", "Nott'm Forest": "Forest", "West Brom": "WestBrom",
    "West Ham": "WestHam", "Sheffield United": "SheffieldUnited", "Crystal Palace": "CrystalPalace",
    "Aston Villa": "AstonVilla",
}


def standardise_team(name: object) -> object:
    """Map any known spelling of a club onto the canonical football-data.co.uk name."""
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


def season_code(start_year: int) -> str:
    """2025 -> '2526' (the football-data.co.uk folder name)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2025 -> '2025/26'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


SEASON_STARTS = list(range(CONFIG["first_season_start"], CONFIG["last_season_start"] + 1))
SEASON_CODES = [season_code(y) for y in SEASON_STARTS]          # ["0001", "0102", ..., "2526", "2627"]
FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
print(f"{len(SEASON_CODES)} season codes: {SEASON_CODES[0]}, {SEASON_CODES[1]}, ..., {SEASON_CODES[-2]}, {SEASON_CODES[-1]}")

CORE_COLS = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee"]
STAT_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
BOOKMAKERS = ["B365", "BW", "IW", "WH", "VC", "LB", "PS", "GB", "SB"]       # individual pre-match books
BOOK_COLS = [f"{b}{o}" for b in BOOKMAKERS for o in "HDA"]
MARKET_COLS = [f"{m}{o}" for m in ("Max", "Avg") for o in "HDA"] + ["Avg>2.5", "Avg<2.5", "B365>2.5", "B365<2.5", "AHh"]
CLOSING_COLS = [f"{b}C{o}" for b in ("B365", "PS", "Avg", "Max") for o in "HDA"]
XG_COLS = ["HxG", "AxG"]
NEEDED_COLS = CORE_COLS + STAT_COLS + BOOK_COLS + MARKET_COLS + CLOSING_COLS + XG_COLS
# Before 2019/20 the market average and maximum were published under Betbrain names.
LEGACY_RENAMES = {"BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA", "BbAvH": "AvgH", "BbAvD": "AvgD",
                  "BbAvA": "AvgA", "BbAv>2.5": "Avg>2.5", "BbAv<2.5": "Avg<2.5", "BbAHh": "AHh",
                  "HT": "HomeTeam", "AT": "AwayTeam"}
COLUMN_GROUPS = {
    "kickoff time": ["Time"], "match stats": STAT_COLS, "Bet365 odds": ["B365H", "B365D", "B365A"],
    "Pinnacle odds": ["PSH", "PSD", "PSA"], "market average odds": ["AvgH", "AvgD", "AvgA"],
    "market max odds": ["MaxH", "MaxD", "MaxA"], "over/under 2.5 odds": ["Avg>2.5", "Avg<2.5"],
    "Asian handicap line": ["AHh"], "Bet365 closing": ["B365CH", "B365CD", "B365CA"],
    "Pinnacle closing": ["PSCH", "PSCD", "PSCA"], "average closing": ["AvgCH", "AvgCD", "AvgCA"],
    "xG (football-data)": XG_COLS,
}
REQUIRED_RESULT_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def make_session(config: dict) -> requests.Session:
    """A requests session with a browser-like User-Agent and automatic retries."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"])
    for prefix in ("http://", "https://"):
        session.mount(prefix, HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": config["user_agent"]})
    return session


SESSION = make_session(CONFIG)


def internet_available(timeout: float = 8.0) -> bool:
    """One cheap request decides whether downloads are attempted at all."""
    try:
        response = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=timeout, stream=True,
                                headers={"User-Agent": CONFIG["user_agent"]})
        response.close()
        return response.status_code < 500
    except Exception:
        return False


def parse_football_data_csv(raw: bytes) -> pd.DataFrame:
    """Parse one football-data.co.uk file.

    Older files are latin-1 encoded and some rows carry more fields than the header (stray trailing commas).
    `on_bad_lines="skip"` would silently drop those rows (45 matches in 2003/04 and in 2004/05), so when rows
    go missing the file is re-read with every row truncated or padded to the header width.
    """
    raw = raw.lstrip(b"\xef\xbb\xbf")
    text = raw.decode("latin-1")
    n_data_lines = sum(1 for line in text.splitlines()[1:] if line.strip(", \t\r"))
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(how="all")
    if len(df) < n_data_lines:
        rows = list(csv.reader(io.StringIO(text)))
        width = len(rows[0])
        fixed = [r[:width] + [""] * (width - len(r)) for r in rows[1:] if any(c.strip() for c in r)]
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(rows[0])
        writer.writerows(fixed)
        df = pd.read_csv(io.StringIO(buffer.getvalue())).dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    return df.loc[:, ~df.columns.duplicated()]


def harmonise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns, keep the union of needed columns and fill absent ones with NaN."""
    renames = {old: new for old, new in LEGACY_RENAMES.items() if old in df.columns and new not in df.columns}
    df = df.rename(columns=renames)
    return df.reindex(columns=NEEDED_COLS)


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse day-first dates written as dd/mm/yy or dd/mm/yyyy."""
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(text, format="%d/%m/%y", errors="coerce")
    parsed = parsed.fillna(short)
    remaining = parsed.isna() & values.notna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(text[remaining], dayfirst=True, errors="coerce")
    return parsed


def season_start_from_date(dates: pd.Series) -> pd.Series:
    """A season runs August to July. COVID pushed 2019/20 into July 2020, hence the July cut-off."""
    return np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1)


def find_input_csvs(config: dict) -> pd.DataFrame:
    """Scan attached Kaggle datasets for EPL result files (used only to fill gaps or when offline)."""
    frames = []
    if not Path(config["input_dir"]).exists():
        return pd.DataFrame()
    for path in sorted(glob.glob(str(config["input_dir"] / "**" / "*.csv"), recursive=True)):
        try:
            with open(path, "rb") as handle:
                df = parse_football_data_csv(handle.read())
            df = df.rename(columns={k: v for k, v in LEGACY_RENAMES.items() if k in df.columns and v not in df.columns})
            if not set(REQUIRED_RESULT_COLS).issubset(df.columns):
                continue
            if "Div" in df.columns:
                df = df[df["Div"].astype(str).str.strip() == "E0"]
            elif not ("E0" in Path(path).stem or "premier" in path.lower() or "epl" in path.lower()):
                continue
            if df.empty:
                continue
            df = harmonise_schema(df)
            dates = parse_dates(df["Date"])
            df = df[dates.notna()].copy()
            df["season_start"] = season_start_from_date(dates[dates.notna()])
            df["source_file"] = path
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_raw_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load every season of EPL results.

    Order of preference per season: (a) a complete cached file in `raw_dir`, (b) a fresh download,
    (c) a stale cached file if the download fails, (d) CSVs attached under `/kaggle/input/**`.
    Returns the harmonised raw table and a per-season summary. Raises if no season can be found.
    """
    raw_dir = Path(config["raw_dir"])
    online = internet_available()
    if not online:
        print("Internet looks unavailable: using cached and attached files only.")
    frames, summary = [], []
    for start in SEASON_STARTS:
        code = season_code(start)
        path = raw_dir / f"E0_{code}.csv"
        df, source = None, None
        if path.exists():
            try:
                cached = parse_football_data_csv(path.read_bytes())
                if len(cached) >= config["full_season_rows"]:
                    df, source = cached, "cache"
            except Exception:
                cached = None
        else:
            cached = None
        if df is None and online:
            try:
                response = SESSION.get(FD_URL.format(code=code), timeout=30)
                if response.status_code == 200 and b"HomeTeam" in response.content[:2000]:
                    path.write_bytes(response.content)
                    df, source = parse_football_data_csv(response.content), "download"
                elif response.status_code == 404:
                    source = "not published yet"
            except Exception as exc:
                source = f"download failed ({type(exc).__name__})"
        if df is None and cached is not None and len(cached):
            df, source = cached, "cache (could not refresh)"
        if df is None:
            summary.append({"season": season_label(start), "code": code, "rows": 0, "source": source or "missing"})
            continue
        df = harmonise_schema(df)
        df = df.dropna(subset=["HomeTeam", "AwayTeam"], how="any")
        df["season_start"] = start
        frames.append(df)
        summary.append({"season": season_label(start), "code": code, "rows": len(df), "source": source})

    # Fill gaps (or everything, when offline) from attached Kaggle datasets.
    loaded = {f["season_start"].iloc[0] for f in frames}
    missing = [s for s in SEASON_STARTS if s not in loaded]
    if missing:
        attached = find_input_csvs(config)
        if len(attached):
            attached = attached.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
            for start in missing:
                part = attached[attached["season_start"] == start].drop(columns=["source_file"])
                if len(part):
                    frames.append(part)
                    for row in summary:
                        if row["code"] == season_code(start):
                            row.update({"rows": len(part), "source": "kaggle input"})
    if not frames:
        raise RuntimeError(
            "No Premier League data could be loaded. Fix one of the following and re-run:\n"
            "  1. Turn on Settings > Internet so the season files can be downloaded from football-data.co.uk.\n"
            f"  2. Place the season files (E0_0001.csv ... E0_{SEASON_CODES[-1]}.csv) in {raw_dir}.\n"
            "  3. Attach a Kaggle dataset containing football-data.co.uk style E0 CSVs "
            "(columns Date, HomeTeam, AwayTeam, FTHG, FTAG).")
    raw = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summary)
    summary["complete"] = summary["rows"] >= config["full_season_rows"]
    return raw, summary


raw_matches, season_summary = load_raw_data(CONFIG)
print(f"Raw table shape: {raw_matches.shape}")
show(season_summary.set_index("season").T, "Rows per season (a full season has 380)")
for _, _row in season_summary.iterrows():
    if not _row["complete"]:
        print(f"INCOMPLETE: season {_row['season']} has {_row['rows']} rows (expected 380) [{_row['source']}]")
_n_ok = int((season_summary["rows"] > 0).sum())
log_source("football-data.co.uk", "ok" if _n_ok == len(SEASON_STARTS) else "partial",
           f"{_n_ok}/{len(SEASON_STARTS)} seasons loaded, {len(raw_matches)} rows; "
           f"{season_code(2025)}: {int(season_summary.loc[season_summary.code == season_code(2025), 'rows'].sum())} rows, "
           f"{season_code(2026)}: {int(season_summary.loc[season_summary.code == season_code(2026), 'rows'].sum())} rows")

# %% [markdown]
# ### Column coverage by season
#
# The table shows the share of matches with a value in each column group. The log below it lists the groups
# each season lacks entirely, which is what drives the NaN handling later on.

# %%
def column_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """Share of non-null values per column group and season."""
    rows = {}
    for start, part in raw.groupby("season_start"):
        rows[season_label(start)] = {name: float(part[cols].notna().all(axis=1).mean())
                                     for name, cols in COLUMN_GROUPS.items()}
    return pd.DataFrame(rows).T


coverage = column_coverage(raw_matches)
show(coverage.round(2), "Column coverage (share of matches with data)")
print("\nSchema log (column groups absent per season):")
for _season, _row in coverage.iterrows():
    _absent = [name for name, value in _row.items() if value == 0]
    print(f"  {_season}: {', '.join(_absent) if _absent else 'none'}")

# %% [markdown]
# ## 2. Cleaning and exploratory analysis
#
# Rows without a date, teams or a final score are dropped, dates are parsed day-first in both two-digit and
# four-digit year formats, and matches are sorted chronologically. The target is `y` with 0 = Away win,
# 1 = Draw, 2 = Home win, an ordered scale that the Ranked Probability Score needs. `matchweek` is
# approximated as the larger of the two teams' game numbers within the season.

# %%
NUMERIC_COLS = [c for c in NEEDED_COLS if c not in ("Date", "Time", "HomeTeam", "AwayTeam", "FTR", "HTR", "Referee")]


def add_matchweek(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate matchweek from the order of each team's games within a season."""
    df = df.copy()
    home_n = df.groupby(["season_start", "HomeTeam"]).cumcount()
    away_n = df.groupby(["season_start", "AwayTeam"]).cumcount()
    long = pd.concat([
        pd.DataFrame({"idx": df.index, "season_start": df["season_start"], "team": df["HomeTeam"], "Date": df["Date"]}),
        pd.DataFrame({"idx": df.index, "season_start": df["season_start"], "team": df["AwayTeam"], "Date": df["Date"]}),
    ]).sort_values(["Date", "idx"], kind="mergesort")
    long["n"] = long.groupby(["season_start", "team"]).cumcount() + 1
    df["matchweek"] = long.groupby("idx")["n"].max().reindex(df.index).astype(int)
    del home_n, away_n
    return df


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw results table and add the targets."""
    df = raw.copy()
    df["Date"] = parse_dates(df["Date"])
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["HomeTeam"] = df["HomeTeam"].map(standardise_team)
    df["AwayTeam"] = df["AwayTeam"].map(standardise_team)
    before = len(df)
    df = df.dropna(subset=REQUIRED_RESULT_COLS)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df.sort_values(["Date", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    df["season"] = df["season_start"].map(season_label)
    df["FTHG"], df["FTAG"] = df["FTHG"].astype(int), df["FTAG"].astype(int)
    df["goal_diff"] = df["FTHG"] - df["FTAG"]
    df["y"] = (np.sign(df["goal_diff"]) + 1).astype(int)          # 0 = Away, 1 = Draw, 2 = Home
    df["is_home_win"] = (df["y"] == 2).astype(int)
    df["FTR"] = df["y"].map({0: "A", 1: "D", 2: "H"})
    df = add_matchweek(df)
    df["match_id"] = np.arange(len(df))
    df["is_fixture"] = False
    print(f"Cleaning: kept {len(df)} of {before} rows, {df['HomeTeam'].nunique()} clubs, "
          f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


matches = clean_matches(raw_matches)
_rows_per_season = matches.groupby("season_start").size()
COMPLETED_SEASONS = [s for s in _rows_per_season.index if _rows_per_season[s] >= CONFIG["full_season_rows"]]
LIVE_SEASONS = [s for s in _rows_per_season.index if s > max(COMPLETED_SEASONS)]
assert matches["Date"].is_monotonic_increasing, "matches must be in chronological order"
assert matches["y"].isin([0, 1, 2]).all()
print(f"Completed seasons: {season_label(min(COMPLETED_SEASONS))} to {season_label(max(COMPLETED_SEASONS))} | "
      f"in progress: {[season_label(s) for s in LIVE_SEASONS]} ({int(_rows_per_season[LIVE_SEASONS].sum())} matches played)")

# %% [markdown]
# ### Optional sources
#
# Each optional source is wrapped so that a failure never stops the notebook. Outcomes are collected in the
# Data availability log.
#
# - **Understat xG** (2014/15 onward). `understatapi` is tried first. If it fails, the notebook reads the same
#   league JSON endpoint that the Understat site itself uses. xG is only ever used through rolling features
#   built from earlier matches.
# - **ClubElo** ratings, fetched once per club and joined as the rating on the day before each match.
# - **FPL archive** (`vaastav/Fantasy-Premier-League`): fixture difficulty ratings from 2019/20. These are kept
#   as context columns only. The archive stores a snapshot rather than the value published before each match,
#   so they are deliberately excluded from every model feature set.

# %%
def _understat_via_package(year: int) -> list:
    from understatapi import UnderstatClient
    with UnderstatClient() as client:
        return client.league(league="EPL").get_match_data(season=str(year))


def _understat_direct(year: int) -> list:
    response = SESSION.get(f"https://understat.com/getLeagueData/EPL/{year}", timeout=30,
                           headers={"X-Requested-With": "XMLHttpRequest"})
    response.raise_for_status()
    return response.json()["dates"]


def load_understat(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach match-level xG (`xg_home`, `xg_away`) from Understat, with football-data xG as a gap filler."""
    matches = matches.copy()
    matches["xg_home"], matches["xg_away"] = np.nan, np.nan
    notes, frames, package_ok = [], [], config["use_understat"]
    seasons = [s for s in sorted(matches["season_start"].unique()) if s >= config["xg_first_season"]]
    for start in (seasons if config["use_understat"] else []):
        cache = Path(config["raw_dir"]) / f"understat_{start}.csv"
        part = None
        if cache.exists():
            cached = pd.read_csv(cache)
            if len(cached) >= config["full_season_rows"] and cached["xg_home"].notna().all():
                part = cached
        if part is None:
            rows = None
            if package_ok:
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        rows = pool.submit(_understat_via_package, start).result(timeout=30)
                except Exception as exc:
                    package_ok = False
                    notes.append(f"understatapi failed ({type(exc).__name__}), using the JSON endpoint")
            if rows is None:
                try:
                    rows = _understat_direct(start)
                except Exception as exc:
                    notes.append(f"{season_label(start)} unavailable ({type(exc).__name__})")
                    if cache.exists():
                        part = pd.read_csv(cache)
            if rows:
                part = pd.DataFrame([{
                    "HomeTeam": standardise_team(r["h"]["title"]), "AwayTeam": standardise_team(r["a"]["title"]),
                    "xg_home": pd.to_numeric(r["xG"]["h"], errors="coerce"),
                    "xg_away": pd.to_numeric(r["xG"]["a"], errors="coerce"),
                } for r in rows if r.get("isResult")])
                if len(part):
                    part.to_csv(cache, index=False)
        if part is not None and len(part):
            part = part.assign(season_start=start)
            frames.append(part)
    n_understat = 0
    if frames:
        xg = pd.concat(frames, ignore_index=True).drop_duplicates(["season_start", "HomeTeam", "AwayTeam"])
        unknown = sorted((set(xg["HomeTeam"]) | set(xg["AwayTeam"])) - (set(matches["HomeTeam"]) | set(matches["AwayTeam"])))
        if unknown:
            notes.append(f"unmapped team names: {unknown}")
        merged = matches[["season_start", "HomeTeam", "AwayTeam"]].merge(
            xg, on=["season_start", "HomeTeam", "AwayTeam"], how="left")
        matches["xg_home"], matches["xg_away"] = merged["xg_home"].values, merged["xg_away"].values
        n_understat = int(matches["xg_home"].notna().sum())
    # football-data.co.uk began publishing HxG/AxG in 2026/27: use it where Understat has no value.
    n_before = int(matches["xg_home"].notna().sum())
    matches["xg_home"] = matches["xg_home"].fillna(matches["HxG"])
    matches["xg_away"] = matches["xg_away"].fillna(matches["AxG"])
    n_fd = int(matches["xg_home"].notna().sum()) - n_before
    eligible = int((matches["season_start"] >= config["xg_first_season"]).sum())
    status = "ok" if n_understat >= 0.95 * eligible else ("partial" if n_understat else "skipped")
    log_source("Understat xG", status, f"{n_understat}/{eligible} matches from Understat, {n_fd} filled from "
               f"football-data HxG/AxG" + (f"; {'; '.join(dict.fromkeys(notes))}" if notes else ""))
    return matches


def load_clubelo(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach ClubElo ratings as of the day before each match (`clubelo_home`, `clubelo_away`)."""
    matches = matches.copy()
    matches["clubelo_home"], matches["clubelo_away"] = np.nan, np.nan
    if not config["use_clubelo"]:
        log_source("ClubElo", "skipped", "disabled in CONFIG")
        return matches
    cache_dir = Path(config["raw_dir"]) / "clubelo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    histories, failures, api_down = {}, [], False
    refresh_after = matches["Date"].max() - pd.Timedelta(days=3)
    for team in teams:
        cache = cache_dir / f"{team.replace(' ', '_')}.csv"
        hist = None
        if cache.exists():
            hist = pd.read_csv(cache, parse_dates=["From", "To"])
            plays_now = team in set(matches.loc[matches["season_start"] == matches["season_start"].max(), "HomeTeam"])
            if plays_now and hist["To"].max() < refresh_after and not api_down:
                hist_stale, hist = hist, None
            else:
                hist_stale = None
        else:
            hist_stale = None
        if hist is None and not api_down:
            try:
                name = CLUBELO_NAMES.get(team, team.replace(" ", ""))
                response = requests.get(f"http://api.clubelo.com/{name}", timeout=12,
                                        headers={"User-Agent": config["user_agent"]})
                response.raise_for_status()
                hist = pd.read_csv(io.StringIO(response.text), parse_dates=["From", "To"])
                if hist.empty or "Elo" not in hist.columns:
                    raise ValueError("empty history")
                hist.to_csv(cache, index=False)
            except Exception as exc:
                failures.append(f"{team} ({type(exc).__name__})")
                hist = hist_stale
                if isinstance(exc, (requests.ConnectionError, requests.Timeout)) or \
                        getattr(getattr(exc, "response", None), "status_code", 0) >= 500:
                    api_down = True     # do not hammer an API that is down
        if hist is None and hist_stale is not None:
            hist = hist_stale
        if hist is not None and len(hist):
            histories[team] = hist.sort_values("From")[["From", "Elo"]].dropna()
    if not histories:
        log_source("ClubElo", "skipped", f"API unreachable ({failures[0] if failures else 'no data'}); "
                   "the pipeline uses its own Elo instead")
        return matches
    for side in ("Home", "Away"):
        lookup = matches[["match_id", "Date", f"{side}Team"]].copy()
        lookup["asof"] = lookup["Date"] - pd.Timedelta(days=1)
        parts = []
        for team, part in lookup.groupby(f"{side}Team"):
            if team in histories:
                joined = pd.merge_asof(part.sort_values("asof"), histories[team], left_on="asof", right_on="From")
                parts.append(joined[["match_id", "Elo"]])
        if parts:
            elo = pd.concat(parts).set_index("match_id")["Elo"]
            matches[f"clubelo_{side.lower()}"] = matches["match_id"].map(elo).values
    share = float(matches["clubelo_home"].notna().mean())
    log_source("ClubElo", "ok" if share > 0.95 and not failures else "partial",
               f"{len(histories)}/{len(teams)} club histories, {share:.0%} of matches covered"
               + (f"; failed: {failures[:5]}" if failures else ""))
    return matches


def load_fpl_context(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach FPL fixture difficulty as context columns (never used as model features)."""
    matches = matches.copy()
    matches["fpl_fdr_home"], matches["fpl_fdr_away"] = np.nan, np.nan
    if not config["use_fpl"]:
        log_source("FPL archive", "skipped", "disabled in CONFIG")
        return matches
    base = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{name}.csv"
    frames, failed = [], []
    for start in [s for s in sorted(matches["season_start"].unique()) if s >= 2019]:
        tag = f"{start}-{(start + 1) % 100:02d}"
        cache = Path(config["raw_dir"]) / f"fpl_fdr_{start}.csv"
        try:
            if cache.exists() and start < matches["season_start"].max():
                part = pd.read_csv(cache)
            else:
                fixtures = pd.read_csv(io.StringIO(SESSION.get(base.format(season=tag, name="fixtures"), timeout=20).text))
                teams = pd.read_csv(io.StringIO(SESSION.get(base.format(season=tag, name="teams"), timeout=20).text))
                names = teams.set_index("id")["name"].map(standardise_team)
                part = pd.DataFrame({
                    "HomeTeam": fixtures["team_h"].map(names), "AwayTeam": fixtures["team_a"].map(names),
                    "fpl_fdr_home": fixtures["team_h_difficulty"], "fpl_fdr_away": fixtures["team_a_difficulty"]})
                part.to_csv(cache, index=False)
            frames.append(part.assign(season_start=start))
        except Exception as exc:
            failed.append(f"{season_label(start)} ({type(exc).__name__})")
    if not frames:
        log_source("FPL archive", "skipped", f"fetch failed for every season: {failed[:3]}")
        return matches
    fdr = pd.concat(frames, ignore_index=True).drop_duplicates(["season_start", "HomeTeam", "AwayTeam"])
    merged = matches[["season_start", "HomeTeam", "AwayTeam"]].merge(fdr, on=["season_start", "HomeTeam", "AwayTeam"], how="left")
    matches["fpl_fdr_home"], matches["fpl_fdr_away"] = merged["fpl_fdr_home"].values, merged["fpl_fdr_away"].values
    eligible = int((matches["season_start"] >= 2019).sum())
    log_source("FPL archive", "ok" if not failed else "partial",
               f"fixture difficulty for {int(matches['fpl_fdr_home'].notna().sum())}/{eligible} matches since 2019/20 "
               "(context only, not a model feature)" + (f"; missing: {failed}" if failed else ""))
    return matches


def safe_optional(loader: Callable, matches: pd.DataFrame, name: str, new_cols: List[str]) -> pd.DataFrame:
    """Run an optional loader. Any unexpected error skips the source and keeps the notebook running."""
    try:
        return loader(matches, CONFIG)
    except Exception as exc:
        log_source(name, "skipped", f"unexpected error: {type(exc).__name__}: {exc}")
        matches = matches.copy()
        for col in new_cols:
            if col not in matches.columns:
                matches[col] = np.nan
        return matches


matches = safe_optional(load_understat, matches, "Understat xG", ["xg_home", "xg_away"])
matches = safe_optional(load_clubelo, matches, "ClubElo", ["clubelo_home", "clubelo_away"])
matches = safe_optional(load_fpl_context, matches, "FPL archive", ["fpl_fdr_home", "fpl_fdr_away"])
HAS_XG = bool(matches["xg_home"].notna().sum() > 1000)
HAS_CLUBELO = bool(matches["clubelo_home"].notna().mean() > 0.5)
if len(find_input_csvs(CONFIG)) == 0:
    log_source("Kaggle input datasets", "not used", "no attached EPL result CSVs found (they are only a gap filler)")

# %% [markdown]
# ### Data availability log

# %%
show(pd.DataFrame(AVAILABILITY_LOG), "Data availability log")

# %% [markdown]
# ### Exploratory analysis
#
# Four quick views of the target: overall class balance, the home-advantage trend, goals per game and the
# draw rate by season. Home advantage has drifted down over 25 years, with a visible dip in the
# behind-closed-doors 2020/21 season, which is one reason why validation must respect time order.

# %%
def season_ticks(ax: plt.Axes, seasons: List[int], step: int = 3) -> None:
    """Label a season axis every few seasons."""
    ax.set_xticks(seasons[::step])
    ax.set_xticklabels([season_label(s) for s in seasons[::step]], rotation=45, ha="right")


def plot_eda(matches: pd.DataFrame) -> None:
    """Class balance, home advantage, goals per game and draw rate."""
    done = matches[matches["season_start"].isin(COMPLETED_SEASONS)]
    by_season = done.groupby("season_start").agg(
        home=("y", lambda s: (s == 2).mean()), draw=("y", lambda s: (s == 1).mean()),
        away=("y", lambda s: (s == 0).mean()), home_goals=("FTHG", "mean"), away_goals=("FTAG", "mean"))
    seasons = list(by_season.index)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    shares = done["y"].value_counts(normalize=True).reindex([2, 1, 0])
    labels = ["Home", "Draw", "Away"]
    bars = ax.bar(labels, shares.values, width=0.55, color=[OUTCOME_COLORS[l] for l in labels])
    for bar, value in zip(bars, shares.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.1%}", ha="center", color=INK)
    ax.set_title(f"Class balance ({len(done):,} matches, completed seasons)")
    ax.set_ylabel("Share of matches")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)

    ax = axes[0, 1]
    for key, label in (("home", "Home"), ("draw", "Draw"), ("away", "Away")):
        ax.plot(seasons, by_season[key], color=OUTCOME_COLORS[label], marker="o", label=label)
        ax.text(seasons[-1] + 0.4, by_season[key].iloc[-1], label, color=INK_SOFT, va="center")
    ax.set_title("Result shares by season (home-advantage trend)")
    ax.set_ylabel("Share of matches")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(seasons[0] - 0.5, seasons[-1] + 3)
    ax.legend(loc="upper right", ncol=3)
    season_ticks(ax, seasons)

    ax = axes[1, 0]
    ax.plot(seasons, by_season["home_goals"], color=PALETTE["blue"], marker="o", label="Home goals")
    ax.plot(seasons, by_season["away_goals"], color=PALETTE["aqua"], marker="o", label="Away goals")
    ax.set_title("Goals per game by season")
    ax.set_ylabel("Goals per match")
    ax.set_ylim(0, None)
    ax.legend(loc="lower right", ncol=2)
    season_ticks(ax, seasons)

    ax = axes[1, 1]
    ax.plot(seasons, by_season["draw"], color=PALETTE["orange"], marker="o")
    ax.axhline(done["y"].eq(1).mean(), color=INK_SOFT, linestyle="--", linewidth=1)
    ax.text(seasons[0], done["y"].eq(1).mean() + 0.004, f"overall {done['y'].eq(1).mean():.1%}", color=INK_SOFT)
    ax.set_title("Draw rate by season")
    ax.set_ylabel("Share of matches drawn")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.15, 0.35)
    season_ticks(ax, seasons)
    fig.tight_layout()
    plt.show()


def plot_missing_heatmap(raw: pd.DataFrame) -> None:
    """Share of missing values per column and season."""
    cols = ["Time", "Referee", "HS", "HST", "HF", "HC", "HY", "HR", "B365H", "BWH", "IWH", "WHH", "VCH", "LBH",
            "PSH", "GBH", "SBH", "AvgH", "MaxH", "Avg>2.5", "AHh", "B365CH", "PSCH", "AvgCH", "HxG"]
    missing = raw.groupby("season_start")[cols].agg(lambda s: s.isna().mean())
    missing.index = [season_label(s) for s in missing.index]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.heatmap(missing.T, cmap=SEQ_CMAP, vmin=0, vmax=1, linewidths=1.5, linecolor=SURFACE, ax=ax,
                cbar_kws={"label": "Share of matches missing", "shrink": 0.7})
    ax.set_title("Missing values by column and season (dark = missing)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    plt.show()


plot_eda(matches)
plot_missing_heatmap(raw_matches)

# %% [markdown]
# ## 3. Leakage-safe feature engineering
#
# Every feature for a match may only use information available **before kickoff**. The design enforces that
# structurally:
#
# - Results are reshaped to a long table with one row per team per match. Every rolling, expanding or
#   exponentially weighted statistic applies `shift(1)` first, so a match never sees its own result.
# - Sequential ratings (Elo, league table, head-to-head) are read **before** the match updates them.
# - Early in a season, rolling windows are padded with the team's per-game averages from the previous season
#   (regression to a prior) instead of leaving NaN. A promoted team has no previous season, so it borrows the
#   average of the previous season's bottom three clubs.
# - Market features come from the pre-match (opening or average) odds. Closing odds are reserved for the
#   bookmaker benchmark and never enter a feature set.
#
# The same `build_features` function is used for training rows and for upcoming fixtures. A fixture is simply a
# row with no result, which is also how the leakage test below works.

# %%
OUTCOME_COLS = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
                "HY", "AY", "HR", "AR", "HxG", "AxG", "xg_home", "xg_away", "goal_diff", "y", "is_home_win"]
ROLL_STATS = ["ppg", "gf", "ga", "gd", "shots", "sot", "sota", "corners", "fouls", "cards", "win", "draw", "loss"]
XG_STATS = ["xg", "xga"]
VENUE_STATS = ["ppg", "gf", "ga", "gd", "sot"]
EWM_STATS = ["gf", "ga", "sot"]


def make_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, sorted by team and date."""
    def side(team: str, opp: str, gf: str, ga: str, s: str, sa: str, st: str, sta: str, c: str, f: str,
             y: str, r: str, xg: str, xga: str, is_home: int) -> pd.DataFrame:
        return pd.DataFrame({
            "match_id": matches["match_id"].values, "Date": matches["Date"].values,
            "season_start": matches["season_start"].values, "team": matches[team].values,
            "opponent": matches[opp].values, "is_home": is_home,
            "gf": matches[gf].astype(float).values, "ga": matches[ga].astype(float).values,
            "shots": matches[s].values, "sot": matches[st].values, "sota": matches[sta].values,
            "corners": matches[c].values, "fouls": matches[f].values,
            "cards": (matches[y] + matches[r]).values, "xg": matches[xg].values, "xga": matches[xga].values})
    long = pd.concat([
        side("HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "HF", "HY", "HR", "xg_home", "xg_away", 1),
        side("AwayTeam", "HomeTeam", "FTAG", "FTHG", "AS", "HS", "AST", "HST", "AC", "AF", "AY", "AR", "xg_away", "xg_home", 0),
    ], ignore_index=True)
    long["gd"] = long["gf"] - long["ga"]
    played = long["gf"].notna()
    long["win"] = np.where(played, (long["gd"] > 0).astype(float), np.nan)
    long["draw"] = np.where(played, (long["gd"] == 0).astype(float), np.nan)
    long["loss"] = np.where(played, (long["gd"] < 0).astype(float), np.nan)
    long["ppg"] = 3 * long["win"] + long["draw"]
    return long.sort_values(["team", "Date", "match_id"], kind="mergesort").reset_index(drop=True)


def season_priors(long: pd.DataFrame) -> dict:
    """Previous-season information for every (season, team): per-game averages, final rank and league means."""
    stats = ROLL_STATS + XG_STATS
    overall = long.groupby(["season_start", "team"])[stats].mean()
    by_venue = long.groupby(["season_start", "team", "is_home"])[stats].mean()
    table = long.groupby(["season_start", "team"]).agg(pts=("ppg", "sum"), gd=("gd", "sum"), gf=("gf", "sum"))
    league = long.groupby("season_start").agg(avg=("gf", "mean"))
    league["home_avg"] = long[long["is_home"] == 1].groupby("season_start")["gf"].mean()
    league["away_avg"] = long[long["is_home"] == 0].groupby("season_start")["gf"].mean()
    seasons = sorted(long["season_start"].unique())
    prior_rows, venue_rows, meta_rows, bottom3 = [], [], [], {}
    for season in seasons:
        teams = sorted(long.loc[long["season_start"] == season, "team"].unique())
        prev = season - 1
        if prev in overall.index.get_level_values(0):
            final = table.loc[prev].sort_values(["pts", "gd", "gf"], ascending=False)
            rank = pd.Series(np.arange(1, len(final) + 1), index=final.index)
            bottom3[season] = list(final.index[-3:])
            prev_overall = overall.loc[prev]
            relegated_avg = prev_overall.loc[bottom3[season]].mean()
            block = prev_overall.reindex(teams)
            promoted = ~pd.Index(teams).isin(prev_overall.index)
            block.loc[promoted] = relegated_avg.values
            for is_home in (0, 1):
                prev_venue = by_venue.loc[prev].xs(is_home, level="is_home")
                vblock = prev_venue.reindex(teams)
                vblock.loc[promoted] = prev_venue.reindex(bottom3[season]).mean().values
                vblock = vblock.assign(season_start=season, is_home=is_home, team=teams)
                venue_rows.append(vblock)
            meta = pd.DataFrame({"team": teams, "season_start": season, "promoted": promoted.astype(float),
                                 "prev_rank": rank.reindex(teams).fillna(19.0).values,
                                 "lg_avg": league.loc[prev, "avg"], "lg_home_avg": league.loc[prev, "home_avg"],
                                 "lg_away_avg": league.loc[prev, "away_avg"]})
        else:   # first season in the data: no history at all
            block = pd.DataFrame(np.nan, index=teams, columns=stats)
            for is_home in (0, 1):
                venue_rows.append(block.assign(season_start=season, is_home=is_home, team=teams))
            meta = pd.DataFrame({"team": teams, "season_start": season, "promoted": np.nan, "prev_rank": 10.5,
                                 "lg_avg": np.nan, "lg_home_avg": np.nan, "lg_away_avg": np.nan})
        prior_rows.append(block.assign(season_start=season, team=teams))
        meta_rows.append(meta)
    return {"overall": pd.concat(prior_rows, ignore_index=True), "venue": pd.concat(venue_rows, ignore_index=True),
            "meta": pd.concat(meta_rows, ignore_index=True), "bottom3": bottom3}


def blended_rolling(frame: pd.DataFrame, stats: List[str], window: int, prior: pd.DataFrame) -> pd.DataFrame:
    """Mean of the previous `window` matches within the season, padded with the prior when fewer exist.

    `frame` must be sorted by team, season and date. `prior` is aligned to `frame` row by row.
    With k earlier matches available: value = (sum of last k + (window - k) * prior) / window.
    """
    keys = [frame["team"], frame["season_start"]]
    shifted = frame.groupby(["team", "season_start"], sort=False)[stats].shift(1)     # never the current match
    rolling = shifted.groupby(keys, sort=False).rolling(window, min_periods=1)
    total = rolling.sum().reset_index(level=[0, 1], drop=True).reindex(frame.index).fillna(0.0)
    count = rolling.count().reset_index(level=[0, 1], drop=True).reindex(frame.index)
    blended = (total + (window - count) * prior[stats].values) / window
    plain = total / count.where(count > 0)
    return blended.where(prior[stats].notna().values, plain)


def seasonal_ewm(frame: pd.DataFrame, stat: str, prior: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted average of earlier matches, restarted from the prior each season."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    values = frame[stat].to_numpy(dtype=float)
    group = (frame["team"] + "|" + frame["season_start"].astype(str)).to_numpy()
    out = np.full(len(frame), np.nan)
    state, current = np.nan, None
    for i in range(len(frame)):
        if group[i] != current:
            current, state = group[i], prior[i]
        out[i] = state                         # read before the match updates the state
        if not np.isnan(values[i]):
            state = values[i] if np.isnan(state) else alpha * values[i] + (1.0 - alpha) * state
    return out


def team_level_features(long: pd.DataFrame, priors: dict, config: dict) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """All per-team pre-match features. Returns the long table plus the form and xG column names."""
    long = long.merge(priors["meta"], on=["season_start", "team"], how="left")
    stats = ROLL_STATS + XG_STATS
    prior_all = long[["season_start", "team"]].merge(priors["overall"], on=["season_start", "team"], how="left")[stats]
    prior_ven = long[["season_start", "team", "is_home"]].merge(
        priors["venue"], on=["season_start", "team", "is_home"], how="left")[stats]
    prior_all.index, prior_ven.index = long.index, long.index
    form_cols, xg_cols, new = [], [], {}

    for window in config["form_windows"]:
        rolled = blended_rolling(long, stats, window, prior_all)
        for stat in stats:
            name = f"{stat}_r{window}"
            new[name] = rolled[stat].values
            (xg_cols if stat in XG_STATS else form_cols).append(name)

    for window in config["venue_windows"]:      # home team's last N home games, away team's last N away games
        for is_home in (0, 1):
            mask = long["is_home"] == is_home
            sub = long[mask]
            rolled = blended_rolling(sub, VENUE_STATS, window, prior_ven[mask])
            for stat in VENUE_STATS:
                name = f"venue_{stat}_r{window}"
                if name not in new:
                    new[name] = np.full(len(long), np.nan)
                    form_cols.append(name)
                new[name][mask.values] = rolled[stat].values

    for stat in EWM_STATS + XG_STATS:
        name = f"{stat}_ewm"
        new[name] = seasonal_ewm(long, stat, prior_all[stat].to_numpy(dtype=float), config["ewm_halflife"])
        (xg_cols if stat in XG_STATS else form_cols).append(name)

    # Attack and defence strength: season-to-date goals per game shrunk towards last season, relative to the
    # previous season's league average.
    grouped = long.groupby(["team", "season_start"], sort=False)
    played = grouped["gf"].transform(lambda s: s.shift(1).notna().cumsum()).values
    weight = config["strength_prior_matches"]
    shrunk = {}
    for stat in ("gf", "ga", "ppg"):
        cumulative = grouped[stat].transform(lambda s: s.shift(1).fillna(0.0).cumsum()).values
        prior = prior_all[stat].values
        with np.errstate(invalid="ignore", divide="ignore"):
            shrunk[stat] = np.where(np.isnan(prior), cumulative / np.where(played > 0, played, np.nan),
                                    (cumulative + weight * prior) / (played + weight))
        if stat == "ppg":
            new["season_pts"] = cumulative
    new["att_strength"] = shrunk["gf"] / long["lg_avg"].values
    new["def_strength"] = shrunk["ga"] / long["lg_avg"].values
    new["season_ppg"] = shrunk["ppg"]
    new["season_played"] = played.astype(float)
    form_cols += ["att_strength", "def_strength", "season_ppg", "season_pts", "season_played"]

    # Rest days and fixture congestion (league matches only, since cups are not in this dataset).
    prev_date = long.groupby("team", sort=False)["Date"].shift(1)
    rest = (long["Date"] - prev_date).dt.days.clip(upper=config["rest_cap_days"]).fillna(config["rest_cap_days"])
    new["rest_days"] = rest.values.astype(float)
    congestion = np.zeros(len(long))
    window_ns = np.timedelta64(config["congestion_days"], "D")
    for _, idx in long.groupby("team", sort=False).indices.items():
        dates = long["Date"].values[idx]
        congestion[idx] = np.arange(len(idx)) - np.searchsorted(dates, dates - window_ns, side="left")
    new[f"matches_last_{config['congestion_days']}d"] = congestion
    form_cols += ["rest_days", f"matches_last_{config['congestion_days']}d", "promoted"]

    long = pd.concat([long, pd.DataFrame(new, index=long.index)], axis=1)
    return long, form_cols, xg_cols


def league_positions(matches: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """League position of each side before kickoff (ties broken by last season's rank)."""
    prev_rank = priors["meta"].set_index(["season_start", "team"])["prev_rank"].to_dict()
    home_pos, away_pos = np.zeros(len(matches)), np.zeros(len(matches))
    seasons, dates = matches["season_start"].to_numpy(), matches["Date"].to_numpy()
    homes, aways = matches["HomeTeam"].to_numpy(), matches["AwayTeam"].to_numpy()
    hgs, ags = matches["FTHG"].to_numpy(dtype=float), matches["FTAG"].to_numpy(dtype=float)
    season_teams = {s: sorted(set(homes[seasons == s]) | set(aways[seasons == s])) for s in np.unique(seasons)}
    i, n = 0, len(matches)
    season, pts, gd, gf = None, {}, {}, {}
    while i < n:                                             # matches are in chronological order
        if seasons[i] != season:
            season = seasons[i]
            teams = season_teams[season]
            pts, gd, gf = dict.fromkeys(teams, 0.0), dict.fromkeys(teams, 0.0), dict.fromkeys(teams, 0.0)
        j = i
        while j < n and dates[j] == dates[i] and seasons[j] == season:
            j += 1
        order = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t], prev_rank.get((season, t), 19.0), t))
        position = {team: k + 1 for k, team in enumerate(order)}
        for r in range(i, j):
            home_pos[r], away_pos[r] = position[homes[r]], position[aways[r]]
        for r in range(i, j):                                # update only after the whole day is recorded
            if np.isnan(hgs[r]) or np.isnan(ags[r]):
                continue
            diff = hgs[r] - ags[r]
            pts[homes[r]] += 3 if diff > 0 else (1 if diff == 0 else 0)
            pts[aways[r]] += 3 if diff < 0 else (1 if diff == 0 else 0)
            gd[homes[r]] += diff
            gd[aways[r]] -= diff
            gf[homes[r]] += hgs[r]
            gf[aways[r]] += ags[r]
        i = j
    return pd.DataFrame({"home_position": home_pos, "away_position": away_pos}, index=matches.index)


def elo_expectation(elo_home: np.ndarray, elo_away: np.ndarray, home_adv: float) -> np.ndarray:
    """Elo win expectancy for the home side (a draw counts as half a win)."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_home + home_adv - elo_away) / 400.0))


def compute_own_elo(matches: pd.DataFrame, priors: dict, config: dict) -> pd.DataFrame:
    """Elo ratings from results only: K-factor, home advantage and a goal-difference multiplier.

    A club absent from the previous season enters at the mean rating of that season's bottom three.
    """
    ratings: Dict[str, float] = {}
    last_seen: Dict[str, int] = {}
    elo_home, elo_away = np.zeros(len(matches)), np.zeros(len(matches))
    cols = matches[["season_start", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].to_numpy(dtype=object)
    for i, (season, home, away, hg, ag) in enumerate(cols):
        for team in (home, away):
            if team not in ratings or last_seen[team] < season - 1:
                relegated = [ratings[t] for t in priors["bottom3"].get(season, []) if t in ratings]
                ratings[team] = float(np.mean(relegated)) if relegated else config["elo_init"]
            last_seen[team] = season
        elo_home[i], elo_away[i] = ratings[home], ratings[away]        # read before updating
        if pd.isna(hg) or pd.isna(ag):
            continue
        expected = 1.0 / (1.0 + 10.0 ** (-(ratings[home] + config["elo_home_adv"] - ratings[away]) / 400.0))
        margin = abs(hg - ag)
        multiplier = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11.0 + margin) / 8.0)
        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = config["elo_k"] * multiplier * (actual - expected)
        ratings[home] += delta
        ratings[away] -= delta
    out = pd.DataFrame({"elo_home": elo_home, "elo_away": elo_away}, index=matches.index)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["elo_exp_home"] = elo_expectation(elo_home, elo_away, config["elo_home_adv"])
    return out


def head_to_head(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Points and goals from the last N meetings of the two clubs (either venue), before this match."""
    history: Dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=config["h2h_matches"]))
    out = np.full((len(matches), 5), np.nan)
    cols = matches[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].to_numpy(dtype=object)
    for i, (home, away, hg, ag) in enumerate(cols):
        past = history[frozenset((home, away))]
        out[i, 0] = len(past)
        if past:
            home_goals = [g_for if team == home else g_against for team, g_for, g_against in past]
            away_goals = [g_against if team == home else g_for for team, g_for, g_against in past]
            home_pts = [3 if h > a else (1 if h == a else 0) for h, a in zip(home_goals, away_goals)]
            away_pts = [3 if a > h else (1 if h == a else 0) for h, a in zip(home_goals, away_goals)]
            out[i, 1:] = [np.mean(home_pts), np.mean(away_pts), np.mean(home_goals), np.mean(away_goals)]
        if not (pd.isna(hg) or pd.isna(ag)):
            past.append((home, hg, ag))                     # stored from the home side's point of view
    return pd.DataFrame(out, index=matches.index,
                        columns=["h2h_matches", "h2h_home_ppg", "h2h_away_ppg", "h2h_home_goals", "h2h_away_goals"])


def market_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Pre-match market features plus the bookmaker benchmark and the odds used in the betting backtest."""
    out = pd.DataFrame(index=matches.index)
    books = {o: matches[[f"{b}{o}" for b in BOOKMAKERS]] for o in "HDA"}
    avg = {o: matches[f"Avg{o}"].fillna(books[o].mean(axis=1)) for o in "HDA"}
    best = {o: matches[f"Max{o}"].fillna(books[o].max(axis=1)) for o in "HDA"}
    inverse = pd.DataFrame({o: 1.0 / avg[o] for o in "HDA"})
    total = inverse.sum(axis=1, min_count=3)
    out["mkt_p_home"], out["mkt_p_draw"], out["mkt_p_away"] = inverse["H"] / total, inverse["D"] / total, inverse["A"] / total
    out["mkt_margin"] = total - 1.0
    out["mkt_p_diff"] = out["mkt_p_home"] - out["mkt_p_away"]
    for o, name in zip("HDA", ("home", "draw", "away")):
        out[f"mkt_spread_{name}"] = best[o] / avg[o] - 1.0          # disagreement across bookmakers
        out[f"open_odds_{name}"] = avg[o]
        out[f"best_odds_{name}"] = best[o]
    over = (1.0 / matches["Avg>2.5"]).fillna(1.0 / matches["B365>2.5"])
    under = (1.0 / matches["Avg<2.5"]).fillna(1.0 / matches["B365<2.5"])
    out["mkt_p_over25"] = over / (over + under)
    out["mkt_ah_line"] = matches["AHh"]
    # Bookmaker benchmark: closing odds where they exist, otherwise the best pre-match odds available.
    source = pd.Series("none", index=matches.index)
    bench = pd.DataFrame(np.nan, index=matches.index, columns=["A", "D", "H"])
    candidates = [("average closing", [f"AvgC{o}" for o in "ADH"]), ("Pinnacle closing", [f"PSC{o}" for o in "ADH"]),
                  ("Bet365 closing", [f"B365C{o}" for o in "ADH"])]
    for label, cols in candidates:
        ok = matches[cols].notna().all(axis=1) & bench["H"].isna()
        bench.loc[ok] = (1.0 / matches.loc[ok, cols]).values
        source[ok] = label
    ok = bench["H"].isna() & total.notna()
    bench.loc[ok] = inverse.loc[ok, ["A", "D", "H"]].values
    source[ok] = "pre-match average"
    bench = bench.div(bench.sum(axis=1), axis=0)
    out["bk_p_away"], out["bk_p_draw"], out["bk_p_home"], out["bk_source"] = bench["A"], bench["D"], bench["H"], source
    return out


MARKET_FEATURES = ["mkt_p_home", "mkt_p_draw", "mkt_p_away", "mkt_p_diff", "mkt_margin", "mkt_spread_home",
                   "mkt_spread_draw", "mkt_spread_away", "mkt_p_over25", "mkt_ah_line"]
META_COLS = ["match_id", "Date", "season_start", "season", "HomeTeam", "AwayTeam", "is_fixture", "FTHG", "FTAG",
             "goal_diff", "y"]
NON_FEATURE_MARKET = ["open_odds_home", "open_odds_draw", "open_odds_away", "best_odds_home", "best_odds_draw",
                      "best_odds_away", "bk_p_away", "bk_p_draw", "bk_p_home", "bk_source"]


def build_features(matches: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build the full pre-match feature table. Rows without a result (fixtures) are supported.

    Returns the table (meta columns, features, benchmark columns) and the feature names per group.
    """
    matches = matches.sort_values(["Date", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    long = make_long(matches)
    priors = season_priors(long)
    long, form_cols, xg_cols = team_level_features(long, priors, config)
    team_cols = form_cols + xg_cols
    home = long[long["is_home"] == 1].set_index("match_id")[team_cols].reindex(matches["match_id"])
    away = long[long["is_home"] == 0].set_index("match_id")[team_cols].reindex(matches["match_id"])
    wide = pd.concat([
        home.add_prefix("home_").reset_index(drop=True), away.add_prefix("away_").reset_index(drop=True),
        pd.DataFrame(home.values - away.values, columns=[f"diff_{c}" for c in team_cols]),
    ], axis=1)
    lg = long[long["is_home"] == 1].set_index("match_id")[["lg_home_avg", "lg_away_avg"]].reindex(matches["match_id"])
    wide["exp_goals_home"] = wide["home_att_strength"].values * wide["away_def_strength"].values * lg["lg_home_avg"].values
    wide["exp_goals_away"] = wide["away_att_strength"].values * wide["home_def_strength"].values * lg["lg_away_avg"].values
    wide["matchweek"] = np.maximum(wide["home_season_played"], wide["away_season_played"]) + 1.0
    position = league_positions(matches, priors)
    position["diff_position"] = position["home_position"] - position["away_position"]
    h2h = head_to_head(matches, config)
    elo = compute_own_elo(matches, priors, config)
    if matches["clubelo_home"].notna().any():
        elo["clubelo_home"], elo["clubelo_away"] = matches["clubelo_home"].values, matches["clubelo_away"].values
        elo["clubelo_diff"] = elo["clubelo_home"] - elo["clubelo_away"]
        elo["clubelo_exp_home"] = elo_expectation(elo["clubelo_home"].values, elo["clubelo_away"].values, config["elo_home_adv"])
    market = market_features(matches)

    def expand(cols: List[str]) -> List[str]:
        return [f"{prefix}_{c}" for prefix in ("home", "away", "diff") for c in cols]

    groups = {
        "form": expand(form_cols) + ["exp_goals_home", "exp_goals_away", "matchweek"] + list(position.columns)
                + list(h2h.columns),
        "elo": list(elo.columns),
        "market": list(MARKET_FEATURES),
        "xg": expand(xg_cols),
    }
    meta = matches[[c for c in META_COLS if c in matches.columns]]
    features = pd.concat([meta, wide, position, h2h, elo, market], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    return features, groups


_t0 = time.time()
features, FEATURE_GROUPS = build_features(matches, CONFIG)
FEATURE_SETS: Dict[str, List[str]] = {
    "A_form_only": FEATURE_GROUPS["form"],
    "B_form_plus_elo": FEATURE_GROUPS["form"] + FEATURE_GROUPS["elo"],
    "C_with_market": FEATURE_GROUPS["form"] + FEATURE_GROUPS["elo"] + FEATURE_GROUPS["market"],
}
XG_FEATURES = FEATURE_GROUPS["xg"]
print(f"Feature table: {features.shape} built in {time.time() - _t0:.1f}s")
for _name, _cols in {**FEATURE_SETS, "xG block (separate experiment)": XG_FEATURES}.items():
    print(f"  {_name}: {len(_cols)} features")
show(features[["Date", "HomeTeam", "AwayTeam", "home_ppg_r5", "away_ppg_r5", "diff_gd_r10", "elo_diff",
               "home_position", "away_position", "mkt_p_home", "y"]].tail(5), "Sample of the feature table")

# %% [markdown]
# ### Leakage tests
#
# Three automated checks run before any model is trained.
#
# 1. **Names and correlation.** No feature is a current-match outcome column, a closing-odds column or a
#    benchmark column, and no feature is suspiciously correlated with the current match's goals or shots.
# 2. **Independent recomputation.** For a random sample of matches, several features are recomputed from scratch
#    with a deliberately naive routine that filters the raw results to strictly earlier dates. The stored values
#    must match.
# 3. **Blank-the-future invariance.** For random cut-off dates, every result on or after the cut-off is erased
#    (those rows become fixtures) and the whole feature table is rebuilt. The features of every match up to and
#    including the cut-off date must be unchanged, which proves that no feature depends on its own match or on
#    anything later.

# %%
ALL_FEATURES = sorted(set(FEATURE_SETS["C_with_market"]) | set(XG_FEATURES))
BLANK_ON_FIXTURE = [c for c in OUTCOME_COLS if c in matches.columns]


def test_feature_names(features: pd.DataFrame) -> None:
    """Static checks on the feature lists."""
    forbidden = set(OUTCOME_COLS) | set(CLOSING_COLS) | set(NON_FEATURE_MARKET) | set(NEEDED_COLS) \
        | {"fpl_fdr_home", "fpl_fdr_away"}
    leaked = forbidden & set(ALL_FEATURES)
    assert not leaked, f"outcome or closing columns used as features: {leaked}"
    assert len(set(ALL_FEATURES)) == len(ALL_FEATURES)
    done = features[~features["is_fixture"]]
    outcome_now = matches.set_index("match_id").loc[done["match_id"], ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]]
    outcome_now["goal_diff"] = outcome_now["FTHG"] - outcome_now["FTAG"]
    worst = 0.0
    for col in outcome_now.columns:
        corr = done[ALL_FEATURES].corrwith(pd.Series(outcome_now[col].values, index=done.index)).abs().max()
        worst = max(worst, float(corr))
    assert worst < 0.75, f"a feature correlates {worst:.2f} with a current-match outcome"
    print(f"1. Name check passed for {len(ALL_FEATURES)} features; highest |correlation| with a current-match "
          f"outcome stat is {worst:.2f}")


def test_recompute_from_history(features: pd.DataFrame, matches: pd.DataFrame, n_sample: int = 250) -> None:
    """Recompute a handful of features naively from strictly earlier matches."""
    results = matches[~matches["is_fixture"]]
    eligible = features[(features["home_season_played"] >= 10) & (features["away_season_played"] >= 10)
                        & ~features["is_fixture"]]
    sample = eligible.sample(n=min(n_sample, len(eligible)), random_state=CONFIG["seed"])
    checked = 0
    for row in sample.itertuples(index=False):
        earlier = results[(results["Date"] < row.Date) & (results["season_start"] == row.season_start)]

        def team_games(team: str) -> pd.DataFrame:
            games = earlier[(earlier["HomeTeam"] == team) | (earlier["AwayTeam"] == team)]
            is_home = games["HomeTeam"] == team
            gf = np.where(is_home, games["FTHG"], games["FTAG"])
            ga = np.where(is_home, games["FTAG"], games["FTHG"])
            return pd.DataFrame({"is_home": is_home.values, "gf": gf, "ga": ga,
                                 "sot": np.where(is_home, games["HST"], games["AST"]),
                                 "pts": np.where(gf > ga, 3, np.where(gf == ga, 1, 0))})
        home, away = team_games(row.HomeTeam), team_games(row.AwayTeam)
        expected = {
            "home_ppg_r5": home["pts"].tail(5).mean(), "away_gf_r10": away["gf"].tail(10).mean(),
            "away_sot_r5": away["sot"].tail(5).mean(), "home_gd_r3": (home["gf"] - home["ga"]).tail(3).mean(),
            "home_season_pts": home["pts"].sum(), "away_season_played": len(away),
        }
        if home["is_home"].sum() >= 5:
            expected["home_venue_gf_r5"] = home.loc[home["is_home"], "gf"].tail(5).mean()
        all_earlier = results[results["Date"] < row.Date]
        last_home = all_earlier[(all_earlier["HomeTeam"] == row.HomeTeam) | (all_earlier["AwayTeam"] == row.HomeTeam)]["Date"].max()
        expected["home_rest_days"] = min((row.Date - last_home).days, CONFIG["rest_cap_days"])
        meetings = all_earlier[((all_earlier["HomeTeam"] == row.HomeTeam) & (all_earlier["AwayTeam"] == row.AwayTeam))
                               | ((all_earlier["HomeTeam"] == row.AwayTeam) & (all_earlier["AwayTeam"] == row.HomeTeam))].tail(5)
        if len(meetings):
            expected["h2h_home_goals"] = np.where(meetings["HomeTeam"] == row.HomeTeam, meetings["FTHG"], meetings["FTAG"]).mean()
        for name, value in expected.items():
            stored = getattr(row, name)
            assert np.isclose(stored, value, atol=1e-9), \
                f"{name} mismatch for {row.HomeTeam} v {row.AwayTeam} on {row.Date.date()}: stored {stored}, recomputed {value}"
            checked += 1
    print(f"2. Independent recomputation passed: {checked} feature values across {len(sample)} random matches")


def test_blank_future_invariance(features: pd.DataFrame, matches: pd.DataFrame, n_cuts: int = 3) -> None:
    """Erase all results from a cut-off date onward and confirm earlier features do not move."""
    dates = matches.loc[matches["season_start"] > matches["season_start"].min() + 1, "Date"].drop_duplicates()
    cuts = sorted(dates.sample(n=n_cuts, random_state=CONFIG["seed"]).tolist())
    cuts.append(matches.loc[matches["season_start"] == max(COMPLETED_SEASONS), "Date"].min())   # an opening day
    compared = 0
    for cut in cuts:
        blanked = matches.copy()
        future = blanked["Date"] >= cut
        blanked.loc[future, BLANK_ON_FIXTURE] = np.nan
        blanked.loc[future, "is_fixture"] = True
        rebuilt, _ = build_features(blanked, CONFIG)
        keep = (features["Date"] <= cut).values
        a = features.loc[keep, ALL_FEATURES].to_numpy(dtype=float)
        b = rebuilt.loc[(rebuilt["Date"] <= cut).values, ALL_FEATURES].to_numpy(dtype=float)
        assert a.shape == b.shape
        same = np.isclose(a, b, atol=1e-9, equal_nan=True)
        if not same.all():
            bad = [ALL_FEATURES[j] for j in np.where(~same.all(axis=0))[0]][:8]
            raise AssertionError(f"future leakage at cut-off {pd.Timestamp(cut).date()}: {bad}")
        compared += a.size
    print(f"3. Blank-the-future invariance passed at {len(cuts)} cut-off dates "
          f"({', '.join(str(pd.Timestamp(c).date()) for c in cuts)}); {compared:,} values compared")


test_feature_names(features)
test_recompute_from_history(features, matches)
test_blank_future_invariance(features, matches)
print("All leakage tests passed.")

# %% [markdown]
# ## 4. Validation design
#
# Football results are a time series: tactics, home advantage and the betting market all drift. Every split is
# therefore chronological and **nothing is shuffled**.
#
# - **Burn-in:** 2000/01 only feeds the rolling features and priors.
# - **Development seasons:** everything up to the holdout. Model selection uses expanding-window walk-forward
#   CV grouped by season: train through 2015/16 and validate 2016/17, train through 2016/17 and validate
#   2017/18, and so on.
# - **Holdout:** the two most recent completed seasons. It is touched once, for the final numbers.
# - **Live block:** completed matches of the season in progress, reported separately because it is small.
#
# The first two walk-forward folds are warm-up folds: their out-of-fold predictions are needed to fit the
# probability calibrators, so CV scores for **every** model are reported on the remaining folds to keep the
# rows comparable. Imputers and scalers live inside each `Pipeline` and are fitted on the training fold only.

# %%
_burn_last = SEASON_STARTS[0] + CONFIG["burn_in_seasons"] - 1
DATA = features[(~features["is_fixture"]) & (features["season_start"] > _burn_last)].reset_index(drop=True)
Y = DATA["y"].astype(int).to_numpy()
SEASON = DATA["season_start"].to_numpy()
GOALS_H, GOALS_A = DATA["FTHG"].to_numpy(dtype=float), DATA["FTAG"].to_numpy(dtype=float)

_completed = [s for s in COMPLETED_SEASONS if s > _burn_last]
TEST_SEASONS = _completed[-CONFIG["n_test_seasons"]:]
DEV_SEASONS = _completed[:-CONFIG["n_test_seasons"]]
CV_VAL_SEASONS = [s for s in DEV_SEASONS if s >= CONFIG["cv_first_val_season"]]
CV_SCORED = CV_VAL_SEASONS[CONFIG["cv_warmup_folds"]:]
IDX_DEV = np.where(np.isin(SEASON, DEV_SEASONS))[0]
IDX_TEST = np.where(np.isin(SEASON, TEST_SEASONS))[0]
IDX_LIVE = np.where(np.isin(SEASON, LIVE_SEASONS))[0]
MASK_CV = np.isin(SEASON, CV_SCORED)
MASK_TEST = np.isin(SEASON, TEST_SEASONS)
MASK_LIVE = np.isin(SEASON, LIVE_SEASONS)
BLOCKS = {"cv": MASK_CV, "test": MASK_TEST, "live": MASK_LIVE}


def walk_forward_splits(val_seasons: List[int], min_season: Optional[int] = None):
    """Expanding-window splits grouped by season. Yields (validation season, train index, validation index)."""
    for season in val_seasons:
        train = (SEASON < season) & np.isin(SEASON, DEV_SEASONS)
        if min_season is not None:
            train &= SEASON >= min_season
        train_idx, val_idx = np.where(train)[0], np.where(SEASON == season)[0]
        assert DATA["Date"].iloc[train_idx].max() < DATA["Date"].iloc[val_idx].min(), "split is not chronological"
        yield season, train_idx, val_idx


assert DATA["Date"].is_monotonic_increasing
assert DATA["Date"].iloc[IDX_DEV].max() < DATA["Date"].iloc[IDX_TEST].min(), "holdout must follow development data"
if len(IDX_LIVE):
    assert DATA["Date"].iloc[IDX_TEST].max() < DATA["Date"].iloc[IDX_LIVE].min()
for _season, _tr, _va in walk_forward_splits(CV_VAL_SEASONS):
    pass   # the generator asserts chronological order for every fold
show(pd.DataFrame([
    {"block": "burn-in (features only)", "seasons": season_label(SEASON_STARTS[0]), "matches": int((features["season_start"] <= _burn_last).sum())},
    {"block": "development", "seasons": f"{season_label(DEV_SEASONS[0])} to {season_label(DEV_SEASONS[-1])}", "matches": len(IDX_DEV)},
    {"block": "  walk-forward warm-up folds", "seasons": ", ".join(season_label(s) for s in CV_VAL_SEASONS[:CONFIG['cv_warmup_folds']]), "matches": int(np.isin(SEASON, CV_VAL_SEASONS[:CONFIG['cv_warmup_folds']]).sum())},
    {"block": "  walk-forward scored folds", "seasons": f"{season_label(CV_SCORED[0])} to {season_label(CV_SCORED[-1])}", "matches": int(MASK_CV.sum())},
    {"block": "holdout (test)", "seasons": ", ".join(season_label(s) for s in TEST_SEASONS), "matches": len(IDX_TEST)},
    {"block": "live (in progress)", "seasons": ", ".join(season_label(s) for s in LIVE_SEASONS) or "none", "matches": len(IDX_LIVE)},
]), "Validation blocks")

# %% [markdown]
# ### Metrics
#
# Log loss is the primary metric. The Ranked Probability Score (RPS) respects the ordering Away < Draw < Home,
# so predicting a draw when the home side wins is penalised less than predicting an away win. The multiclass
# Brier score, accuracy, macro-F1 and per-class precision and recall complete the picture. Draws get their own
# columns because almost every model under-predicts them.

# %%
def per_match_log_loss(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Negative log probability assigned to the observed outcome."""
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def per_match_rps(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked Probability Score for ordered outcomes (0 = Away, 1 = Draw, 2 = Home). Lower is better."""
    observed = np.eye(P.shape[1])[y]
    cumulative_gap = np.cumsum(P, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]
    return (cumulative_gap ** 2).sum(axis=1) / (P.shape[1] - 1)


def per_match_brier(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiclass Brier score: squared error summed over the three classes."""
    return ((P - np.eye(P.shape[1])[y]) ** 2).sum(axis=1)


def prob_metrics(P: np.ndarray, y: np.ndarray, hard: bool = False) -> dict:
    """All classification metrics for one block of matches."""
    pred = P.argmax(axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    out = {"logloss": np.nan if hard else float(per_match_log_loss(P, y).mean()),
           "rps": float(per_match_rps(P, y).mean()), "brier": float(per_match_brier(P, y).mean()),
           "acc": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))}
    for k, name in enumerate(("away", "draw", "home")):
        out[f"prec_{name}"], out[f"rec_{name}"] = float(precision[k]), float(recall[k])
    return out


PREDS: Dict[Tuple[str, str], np.ndarray] = {}      # (model, feature set) -> probabilities for every DATA row
FITTED: Dict[Tuple[str, str], object] = {}         # models trained on the development seasons
FIT_FNS: Dict[Tuple[str, str], Callable] = {}      # how to refit each model on more data
RESULTS: Dict[Tuple[str, str], dict] = {}
NOTES: Dict[Tuple[str, str], str] = {}


def register(model: str, fset: str, P_all: np.ndarray, hard: bool = False, note: str = "") -> None:
    """Score one model on the CV, holdout and live blocks and store it in the comparison registry."""
    row = {"model": model, "feature_set": fset}
    for block, mask in BLOCKS.items():
        if mask.sum() == 0:
            continue
        P = P_all[mask]
        if np.isnan(P).any():           # for example hard goal-difference predictions with no probabilities
            continue
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), f"{model}: probabilities must sum to 1"
        assert (P >= 0).all()
        row.update({f"{block}_{k}": v for k, v in prob_metrics(P, Y[mask], hard=hard).items()})
        row[f"{block}_n"] = int(mask.sum())
    PREDS[(model, fset)] = P_all
    RESULTS[(model, fset)] = row
    NOTES[(model, fset)] = note


def comparison_table() -> pd.DataFrame:
    """Every registered model, best walk-forward log loss first."""
    table = pd.DataFrame(list(RESULTS.values()))
    table["note"] = [NOTES[key] for key in RESULTS]
    return table.sort_values(["cv_logloss", "cv_rps"], na_position="last").reset_index(drop=True)


SHOW_COLS = ["model", "feature_set", "cv_logloss", "cv_rps", "cv_acc", "test_logloss", "test_rps", "test_brier",
             "test_acc", "test_macro_f1", "test_rec_draw", "live_logloss", "live_acc"]


def show_comparison(title: str) -> None:
    """Print the running comparison table."""
    table = comparison_table()
    show(table[[c for c in SHOW_COLS if c in table.columns]], f"{title} (sorted by walk-forward log loss; "
         f"elapsed {(time.time() - NOTEBOOK_T0) / 60:.1f} min)")


def make_pipeline(estimator: object, scale: bool) -> Pipeline:
    """Median imputation (and optional standardisation) fitted inside the training fold, then the estimator."""
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("model", estimator)])


def run_classifier(model: str, fset: str, fit_fn: Callable, cols: Optional[List[str]] = None,
                   val_seasons: Optional[List[int]] = None, min_season: Optional[int] = None,
                   store: bool = True) -> np.ndarray:
    """Walk-forward CV plus a final fit on all development seasons, scored on the holdout and live blocks.

    `fit_fn(X, y, seasons)` returns a fitted object with `predict_proba`.
    """
    cols = cols or FEATURE_SETS[fset]
    X = DATA[cols]
    P_all = np.full((len(DATA), 3), np.nan)
    for _, train_idx, val_idx in walk_forward_splits(val_seasons or CV_VAL_SEASONS, min_season):
        fitted = fit_fn(X.iloc[train_idx], Y[train_idx], SEASON[train_idx])
        P_all[val_idx] = fitted.predict_proba(X.iloc[val_idx])
    dev_idx = IDX_DEV if min_season is None else IDX_DEV[SEASON[IDX_DEV] >= min_season]
    final = fit_fn(X.iloc[dev_idx], Y[dev_idx], SEASON[dev_idx])
    if isinstance(final, Pipeline):
        transformed = final[:-1].transform(X.iloc[IDX_TEST])
        assert not np.isnan(transformed).any(), "NaN reached the model after imputation"
    for idx in (IDX_TEST, IDX_LIVE):
        if len(idx):
            P_all[idx] = final.predict_proba(X.iloc[idx])
    if store:
        register(model, fset, P_all)
        FITTED[(model, fset)], FIT_FNS[(model, fset)] = final, fit_fn
    return P_all


def recent_fold_score(fit_fn: Callable, cols: List[str], n_folds: int = 3) -> float:
    """Mean log loss on the most recent walk-forward folds (used to tune simple hyperparameters)."""
    losses = []
    for _, train_idx, val_idx in walk_forward_splits(CV_VAL_SEASONS[-n_folds:]):
        fitted = fit_fn(DATA[cols].iloc[train_idx], Y[train_idx], SEASON[train_idx])
        losses.append(per_match_log_loss(fitted.predict_proba(DATA[cols].iloc[val_idx]), Y[val_idx]).mean())
    return float(np.mean(losses))

# %% [markdown]
# ## 5. Classification models
#
# Models are trained in order of complexity and each one is run on all three feature sets:
# `A_form_only` (no odds, no Elo), `B_form_plus_elo` (no odds) and `C_with_market` (adds the odds features).
# The comparison table is printed after every model.
#
# ### 5.1 Baselines and the bookmaker benchmark
#
# - **Class prior (majority class):** predicts the training-fold class frequencies, so it always picks Home.
# - **Always home:** a hard prediction with no probabilities, so log loss is undefined for it.
# - **Bookmaker:** margin-removed implied probabilities from average closing odds where they exist (2019/20
#   onward), then Pinnacle closing odds, then the pre-match market average. This is the benchmark to beat, and
#   closing odds contain information (team news, late money) that no pre-match model here can see.

# %%
def baseline_predictions() -> None:
    """Register the two naive baselines and the bookmaker benchmark."""
    prior = np.full((len(DATA), 3), np.nan)
    for _, train_idx, val_idx in walk_forward_splits(CV_VAL_SEASONS):
        prior[val_idx] = np.bincount(Y[train_idx], minlength=3) / len(train_idx)
    prior[MASK_TEST | MASK_LIVE] = np.bincount(Y[IDX_DEV], minlength=3) / len(IDX_DEV)
    register("Class prior (majority class)", "-", prior, note="training-fold class frequencies")

    always_home = np.full((len(DATA), 3), np.nan)
    always_home[np.isin(SEASON, CV_VAL_SEASONS) | MASK_TEST | MASK_LIVE] = [0.0, 0.0, 1.0]
    register("Always home", "-", always_home, hard=True, note="hard prediction, log loss undefined")

    bookmaker = DATA[["bk_p_away", "bk_p_draw", "bk_p_home"]].to_numpy(dtype=float)
    missing = np.isnan(bookmaker).any(axis=1)
    bookmaker[missing] = np.bincount(Y[IDX_DEV], minlength=3) / len(IDX_DEV)
    sources = DATA.loc[MASK_TEST, "bk_source"].value_counts().to_dict()
    register("Bookmaker (closing odds)", "-", bookmaker, note=f"holdout odds source: {sources}")
    print(f"Bookmaker benchmark source on the holdout: {sources}; rows without any odds: {int(missing.sum())}")


baseline_predictions()
show_comparison("After baselines")

# %% [markdown]
# ### 5.2 Multinomial logistic regression
#
# Standardised inputs, L2 penalty. The inverse regularisation strength `C` is tuned per feature set on the most
# recent walk-forward folds. With about 180 correlated features, strong regularisation is expected to win.

# %%
def make_logreg_fit(C: float) -> Callable:
    def fit(X: pd.DataFrame, y: np.ndarray, seasons: np.ndarray) -> Pipeline:
        return make_pipeline(LogisticRegression(C=C, max_iter=500), scale=True).fit(X, y)
    return fit


BEST_C: Dict[str, float] = {}
for _fset, _cols in FEATURE_SETS.items():
    _scores = {C: recent_fold_score(make_logreg_fit(C), _cols) for C in CONFIG["logreg_C_grid"]}
    BEST_C[_fset] = min(_scores, key=_scores.get)
    print(f"{_fset}: C grid log loss {({k: round(v, 4) for k, v in _scores.items()})} -> C = {BEST_C[_fset]}")
    run_classifier("Logistic regression", _fset, make_logreg_fit(BEST_C[_fset]))
    NOTES[("Logistic regression", _fset)] = f"C={BEST_C[_fset]}"
show_comparison("After logistic regression")

# %% [markdown]
# ### 5.3 Random forest
#
# A modest-depth forest with large leaves. It needs no scaling and gives a non-linear reference point before
# boosting.

# %%
def fit_random_forest(X: pd.DataFrame, y: np.ndarray, seasons: np.ndarray) -> Pipeline:
    forest = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=15, max_features="sqrt",
                                    n_jobs=CONFIG["n_jobs"], random_state=CONFIG["seed"])
    return make_pipeline(forest, scale=False).fit(X, y)


for _fset in FEATURE_SETS:
    run_classifier("Random forest", _fset, fit_random_forest)
show_comparison("After random forest")

# %% [markdown]
# ### 5.4 XGBoost with Optuna
#
# `multi:softprob` with shallow trees, a low learning rate, row and column subsampling and strong
# regularisation. The number of boosting rounds comes from early stopping on a **time-ordered** slice: the
# last season of each training fold. The model is then refitted on the whole fold with that many rounds.
#
# Optuna searches the remaining hyperparameters against walk-forward log loss on the most recent folds. The
# search is capped at about 40 trials and time-boxed per feature set. Because the tuned parameters were chosen
# on some of the scored folds, the XGBoost CV scores are slightly optimistic. The holdout is never used.

# %%
XGB_FIXED = {"objective": "multi:softprob", "eval_metric": "mlogloss", "tree_method": "hist", "max_bin": 128,
             "n_jobs": CONFIG["n_jobs"], "random_state": CONFIG["seed"], "verbosity": 0}


def xgb_early_stopping_fit(X_train: np.ndarray, y_train: np.ndarray, X_stop: np.ndarray, y_stop: np.ndarray,
                           params: dict) -> xgb.XGBClassifier:
    """Fit with early stopping on a later, time-ordered slice."""
    model = xgb.XGBClassifier(**XGB_FIXED, **params, n_estimators=CONFIG["xgb_max_rounds"],
                              early_stopping_rounds=CONFIG["xgb_early_stopping"])
    model.fit(X_train, y_train, eval_set=[(X_stop, y_stop)], verbose=False)
    return model


def make_xgb_fit(params: dict, rounds: Optional[int] = None) -> Callable:
    """Fit function: pick the number of rounds on the last training season, then refit on everything."""
    def fit(X: pd.DataFrame, y: np.ndarray, seasons: np.ndarray) -> Pipeline:
        n_rounds = rounds
        if n_rounds is None:
            stop = seasons == seasons.max()
            imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[~stop])
            probe = xgb_early_stopping_fit(imputer.transform(X[~stop]), y[~stop], imputer.transform(X[stop]), y[stop], params)
            n_rounds = int(np.ceil((probe.best_iteration + 1) * 1.1))    # a little more data, a few more rounds
        model = xgb.XGBClassifier(**XGB_FIXED, **params, n_estimators=n_rounds)
        return make_pipeline(model, scale=False).fit(X, y)
    return fit


def tune_xgboost(fset: str) -> Tuple[dict, optuna.study.Study]:
    """Optuna search on walk-forward log loss (most recent folds, early stopping on the preceding season)."""
    cols = FEATURE_SETS[fset]
    folds = []
    for season in CV_VAL_SEASONS[-CONFIG["optuna_folds"]:]:
        train = np.where((SEASON < season - 1) & np.isin(SEASON, DEV_SEASONS))[0]
        stop, val = np.where(SEASON == season - 1)[0], np.where(SEASON == season)[0]
        assert DATA["Date"].iloc[train].max() < DATA["Date"].iloc[stop].min() < DATA["Date"].iloc[val].min()
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(DATA[cols].iloc[train])
        folds.append(tuple(imputer.transform(DATA[cols].iloc[i]) for i in (train, stop, val)) + (Y[train], Y[stop], Y[val]))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 40.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
        losses = []
        for X_train, X_stop, X_val, y_train, y_stop, y_val in folds:
            model = xgb_early_stopping_fit(X_train, y_train, X_stop, y_stop, params)
            losses.append(per_match_log_loss(model.predict_proba(X_val), y_val).mean())
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=CONFIG["seed"]))
    study.optimize(objective, n_trials=CONFIG["optuna_trials"], timeout=CONFIG["optuna_timeout_s"])
    return study.best_params, study


XGB_PARAMS: Dict[str, dict] = {}
for _fset in FEATURE_SETS:
    _t0 = time.time()
    XGB_PARAMS[_fset], _study = tune_xgboost(_fset)
    print(f"{_fset}: {len(_study.trials)} Optuna trials in {time.time() - _t0:.0f}s, best log loss "
          f"{_study.best_value:.4f}, params { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in XGB_PARAMS[_fset].items()} }")
    run_classifier("XGBoost", _fset, make_xgb_fit(XGB_PARAMS[_fset]))
    _rounds = FITTED[("XGBoost", _fset)].named_steps["model"].n_estimators
    NOTES[("XGBoost", _fset)] = f"{len(_study.trials)} Optuna trials, {_rounds} rounds"
show_comparison("After XGBoost")

# %% [markdown]
# ### 5.5 LightGBM (optional comparison)
#
# If LightGBM is installed it is run with fixed, conservative parameters as a second boosting implementation.

# %%
def fit_lightgbm(X: pd.DataFrame, y: np.ndarray, seasons: np.ndarray) -> Pipeline:
    model = lgb.LGBMClassifier(objective="multiclass", n_estimators=250, learning_rate=0.03, num_leaves=8,
                               max_depth=4, min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=10.0, random_state=CONFIG["seed"],
                               n_jobs=CONFIG["n_jobs"], verbose=-1)
    return make_pipeline(model, scale=False).fit(X.to_numpy(dtype=float), y)


class _ArrayInput:
    """Thin wrapper so a pipeline fitted on arrays accepts DataFrames without feature-name warnings."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(np.asarray(X, dtype=float))


if HAS_LGBM:
    for _fset in FEATURE_SETS:
        run_classifier("LightGBM", _fset, lambda X, y, s: _ArrayInput(fit_lightgbm(X, y, s)))
    show_comparison("After LightGBM")
else:
    print("LightGBM is not installed: skipped.")

# %% [markdown]
# ### 5.6 Probability calibration
#
# Calibrators are fitted on **walk-forward out-of-fold predictions**, which are genuine out-of-sample
# probabilities from seasons that precede the block being calibrated:
#
# - for a scored CV season, the calibrator sees only out-of-fold predictions from earlier seasons;
# - for the holdout and live blocks, it sees the out-of-fold predictions from all walk-forward seasons.
#
# Two methods are compared: isotonic regression (one-vs-rest, renormalised) and sigmoid calibration (a
# multinomial Platt scaling on log-probabilities). The method is chosen per model on walk-forward log loss.

# %%
def fit_calibrator(P: np.ndarray, y: np.ndarray, method: str) -> dict:
    """Fit an isotonic or sigmoid calibrator on out-of-sample probabilities."""
    if method == "isotonic":
        models = [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(P[:, k], (y == k).astype(float))
                  for k in range(P.shape[1])]
        return {"method": method, "models": models}
    model = LogisticRegression(C=100.0, max_iter=1000).fit(np.log(np.clip(P, 1e-6, 1.0)), y)
    return {"method": method, "model": model}


def apply_calibrator(calibrator: dict, P: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator and renormalise so each row sums to 1."""
    if calibrator["method"] == "isotonic":
        out = np.column_stack([m.predict(P[:, k]) for k, m in enumerate(calibrator["models"])])
    else:
        out = calibrator["model"].predict_proba(np.log(np.clip(P, 1e-6, 1.0)))
    out = np.clip(out, 1e-4, 1.0)
    return out / out.sum(axis=1, keepdims=True)


CALIBRATORS: Dict[Tuple[str, str], dict] = {}


def calibrate_walk_forward(model: str, fset: str) -> None:
    """Calibrate a model's probabilities without ever fitting a calibrator on later data."""
    raw = PREDS[(model, fset)]
    candidates = {}
    for method in ("sigmoid", "isotonic"):
        P_all = np.full_like(raw, np.nan)
        for season in CV_SCORED:
            earlier = np.isin(SEASON, [s for s in CV_VAL_SEASONS if s < season])
            assert DATA["Date"][earlier].max() < DATA["Date"][SEASON == season].min()
            P_all[SEASON == season] = apply_calibrator(fit_calibrator(raw[earlier], Y[earlier], method), raw[SEASON == season])
        fit_mask = np.isin(SEASON, CV_VAL_SEASONS)
        final = fit_calibrator(raw[fit_mask], Y[fit_mask], method)
        later = MASK_TEST | MASK_LIVE
        P_all[later] = apply_calibrator(final, raw[later])
        candidates[method] = (per_match_log_loss(P_all[MASK_CV], Y[MASK_CV]).mean(), P_all, final)
    best = min(candidates, key=lambda m: candidates[m][0])
    name = f"{model} (calibrated)"
    register(name, fset, candidates[best][1], note=f"{best} calibration; walk-forward log loss sigmoid "
             f"{candidates['sigmoid'][0]:.4f}, isotonic {candidates['isotonic'][0]:.4f}")
    CALIBRATORS[(name, fset)] = candidates[best][2]


for _fset in FEATURE_SETS:
    for _model in ("Logistic regression", "XGBoost"):
        calibrate_walk_forward(_model, _fset)
        print(f"{_model} / {_fset}: {NOTES[(f'{_model} (calibrated)', _fset)]}")
show_comparison("After calibration")

# %% [markdown]
# ### 5.7 Ensemble
#
# A weighted average of the calibrated logistic regression and calibrated XGBoost probabilities. The weight is
# chosen on validation data only: for each scored CV season it is picked on the calibrated predictions of
# earlier scored seasons (0.5 for the first one), and for the holdout it is picked on all scored CV seasons.

# %%
ENSEMBLE_WEIGHTS: Dict[str, float] = {}
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


def best_weight(P_lr: np.ndarray, P_xgb: np.ndarray, y: np.ndarray) -> float:
    """Weight on logistic regression that minimises log loss on the given validation rows."""
    losses = [per_match_log_loss(w * P_lr + (1 - w) * P_xgb, y).mean() for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmin(losses))])


for _fset in FEATURE_SETS:
    _lr, _xg = PREDS[("Logistic regression (calibrated)", _fset)], PREDS[("XGBoost (calibrated)", _fset)]
    _ens = np.full_like(_lr, np.nan)
    for _i, _season in enumerate(CV_SCORED):
        _earlier = np.isin(SEASON, CV_SCORED[:_i])
        _w = best_weight(_lr[_earlier], _xg[_earlier], Y[_earlier]) if _i > 0 else 0.5
        _ens[SEASON == _season] = _w * _lr[SEASON == _season] + (1 - _w) * _xg[SEASON == _season]
    ENSEMBLE_WEIGHTS[_fset] = best_weight(_lr[MASK_CV], _xg[MASK_CV], Y[MASK_CV])
    _later = MASK_TEST | MASK_LIVE
    _ens[_later] = ENSEMBLE_WEIGHTS[_fset] * _lr[_later] + (1 - ENSEMBLE_WEIGHTS[_fset]) * _xg[_later]
    register("Ensemble LR + XGBoost (calibrated)", _fset, _ens,
             note=f"weight on logistic regression = {ENSEMBLE_WEIGHTS[_fset]:.1f}")
    print(f"{_fset}: ensemble weight on logistic regression = {ENSEMBLE_WEIGHTS[_fset]:.1f}")
show_comparison("After the ensemble")

# %% [markdown]
# ## 6. Goals regression models
#
# The second route to match probabilities goes through goals.
#
# 1. **Linear regression on goal difference**: the simplest possible baseline, scored by whether the sign of the
#    prediction matches the sign of the result.
# 2. **Poisson GLM** (`PoissonRegressor`) fitted separately for home goals and away goals.
# 3. **XGBoost** with `objective="count:poisson"` for both targets.
#
# The two predicted scoring rates are turned into a 9 x 9 scoreline matrix (0 to 8 goals each). Independent
# Poisson scores under-count low-scoring draws, so the Dixon-Coles correction rescales the 0-0, 1-0, 0-1 and 1-1
# cells with a single parameter `rho`, estimated by maximum likelihood on the training fold. Summing regions
# of the matrix gives P(Home), P(Draw), P(Away), P(over 2.5 goals) and P(both teams to score), and the 1X2
# probabilities are scored with exactly the same metrics as the classifiers.

# %%
def dixon_coles_tau(home_goals: np.ndarray, away_goals: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray,
                    rho: float) -> np.ndarray:
    """Dixon-Coles low-score adjustment factor for each match."""
    tau = np.ones_like(lam_h, dtype=float)
    tau = np.where((home_goals == 0) & (away_goals == 0), 1.0 - lam_h * lam_a * rho, tau)
    tau = np.where((home_goals == 0) & (away_goals == 1), 1.0 + lam_h * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1.0 + lam_a * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, tau)
    return tau


def fit_rho(home_goals: np.ndarray, away_goals: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray) -> float:
    """Maximum-likelihood estimate of the Dixon-Coles dependence parameter given the scoring rates."""
    def negative_log_likelihood(rho: float) -> float:
        return -float(np.log(np.clip(dixon_coles_tau(home_goals, away_goals, lam_h, lam_a, rho), 1e-9, None)).sum())
    return float(minimize_scalar(negative_log_likelihood, bounds=(-0.2, 0.2), method="bounded").x)


def scoreline_grid(lam_h: np.ndarray, lam_a: np.ndarray, rho: float = 0.0, max_goals: int = 8) -> np.ndarray:
    """P(home goals = i, away goals = j) for every match, shape (n, max_goals + 1, max_goals + 1)."""
    goals = np.arange(max_goals + 1)
    grid = poisson.pmf(goals[None, :], lam_h[:, None])[:, :, None] * poisson.pmf(goals[None, :], lam_a[:, None])[:, None, :]
    grid[:, 0, 0] *= 1.0 - lam_h * lam_a * rho
    grid[:, 0, 1] *= 1.0 + lam_h * rho
    grid[:, 1, 0] *= 1.0 + lam_a * rho
    grid[:, 1, 1] *= 1.0 - rho
    grid = np.clip(grid, 0.0, None)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def grid_summary(grid: np.ndarray) -> dict:
    """1X2, over 2.5 and both-teams-to-score probabilities plus the most likely scoreline."""
    n_goals = grid.shape[1]
    i, j = np.indices((n_goals, n_goals))
    p_home, p_draw, p_away = grid[:, i > j].sum(axis=1), grid[:, i == j].sum(axis=1), grid[:, i < j].sum(axis=1)
    flat = grid.reshape(len(grid), -1).argmax(axis=1)
    return {"P": np.column_stack([p_away, p_draw, p_home]), "over25": grid[:, (i + j) >= 3].sum(axis=1),
            "btts": grid[:, (i >= 1) & (j >= 1)].sum(axis=1), "mode_home": flat // n_goals, "mode_away": flat % n_goals}


def goals_predict(bundle: dict, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Predicted scoring rates (home, away) from a fitted goals bundle."""
    X_in = X if bundle.get("frame_input", True) else np.asarray(X, dtype=float)
    return (np.clip(bundle["home"].predict(X_in), 0.05, 6.0), np.clip(bundle["away"].predict(X_in), 0.05, 6.0))


def make_poisson_glm_fit(alpha: float) -> Callable:
    def fit(X: pd.DataFrame, goals_h: np.ndarray, goals_a: np.ndarray, seasons: np.ndarray) -> dict:
        return {"home": make_pipeline(PoissonRegressor(alpha=alpha, max_iter=300), scale=True).fit(X, goals_h),
                "away": make_pipeline(PoissonRegressor(alpha=alpha, max_iter=300), scale=True).fit(X, goals_a)}
    return fit


XGB_POISSON_PARAMS = {"objective": "count:poisson", "eval_metric": "poisson-nloglik", "tree_method": "hist",
                      "max_bin": 128, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
                      "colsample_bytree": 0.5, "min_child_weight": 10, "reg_lambda": 10.0,
                      "n_jobs": CONFIG["n_jobs"], "random_state": CONFIG["seed"], "verbosity": 0}


def fit_xgb_poisson(X: pd.DataFrame, goals_h: np.ndarray, goals_a: np.ndarray, seasons: np.ndarray,
                    rounds: Optional[dict] = None) -> dict:
    """XGBoost Poisson regressors. Rounds come from early stopping on the last training season unless given."""
    bundle = {"rounds": dict(rounds) if rounds else {}}
    if not rounds:
        stop = seasons == seasons.max()
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[~stop])
        X_train, X_stop = imputer.transform(X[~stop]), imputer.transform(X[stop])
    for side, target in (("home", goals_h), ("away", goals_a)):
        if not rounds:
            probe = xgb.XGBRegressor(**XGB_POISSON_PARAMS, n_estimators=600, early_stopping_rounds=30)
            probe.fit(X_train, target[~stop], eval_set=[(X_stop, target[stop])], verbose=False)
            bundle["rounds"][side] = int(np.ceil((probe.best_iteration + 1) * 1.1))
        model = xgb.XGBRegressor(**XGB_POISSON_PARAMS, n_estimators=bundle["rounds"][side])
        bundle[side] = make_pipeline(model, scale=False).fit(X, target)
    return bundle


GOALS: Dict[Tuple[str, str], dict] = {}
GOAL_FIT_FNS: Dict[Tuple[str, str], Callable] = {}
REG_RESULTS: List[dict] = []


def goal_metrics(lam_h: np.ndarray, lam_a: np.ndarray, mask: np.ndarray) -> dict:
    """Regression metrics for predicted scoring rates."""
    gh, ga = GOALS_H[mask], GOALS_A[mask]
    pred_gd, true_gd = lam_h - lam_a, gh - ga
    decisive = true_gd != 0
    return {"mae_home": mean_absolute_error(gh, lam_h), "mae_away": mean_absolute_error(ga, lam_a),
            "rmse_home": float(np.sqrt(mean_squared_error(gh, lam_h))), "rmse_away": float(np.sqrt(mean_squared_error(ga, lam_a))),
            "poisson_dev_home": mean_poisson_deviance(gh, lam_h), "poisson_dev_away": mean_poisson_deviance(ga, lam_a),
            "gd_mae": mean_absolute_error(true_gd, pred_gd), "gd_rmse": float(np.sqrt(mean_squared_error(true_gd, pred_gd))),
            "gd_sign_acc": float((np.sign(pred_gd) == np.sign(true_gd)).mean()),
            "gd_sign_acc_decisive": float((np.sign(pred_gd[decisive]) == np.sign(true_gd[decisive])).mean())}


def run_goals_model(model: str, fset: str, fit_fn: Callable, store: bool = True) -> dict:
    """Walk-forward CV and holdout predictions for a pair of goal models, converted to 1X2 probabilities."""
    X = DATA[FEATURE_SETS[fset]]
    n = len(DATA)
    out = {"lam_h": np.full(n, np.nan), "lam_a": np.full(n, np.nan), "P": np.full((n, 3), np.nan),
           "over25": np.full(n, np.nan), "btts": np.full(n, np.nan)}

    def fit_and_fill(train_idx: np.ndarray, targets: List[np.ndarray]) -> dict:
        bundle = fit_fn(X.iloc[train_idx], GOALS_H[train_idx], GOALS_A[train_idx], SEASON[train_idx])
        fit_h, fit_a = goals_predict(bundle, X.iloc[train_idx])
        bundle["rho"] = fit_rho(GOALS_H[train_idx], GOALS_A[train_idx], fit_h, fit_a)
        for idx in targets:
            if len(idx) == 0:
                continue
            lam_h, lam_a = goals_predict(bundle, X.iloc[idx])
            summary = grid_summary(scoreline_grid(lam_h, lam_a, bundle["rho"], CONFIG["max_goals"]))
            out["lam_h"][idx], out["lam_a"][idx], out["P"][idx] = lam_h, lam_a, summary["P"]
            out["over25"][idx], out["btts"][idx] = summary["over25"], summary["btts"]
        return bundle

    for _, train_idx, val_idx in walk_forward_splits(CV_VAL_SEASONS):
        fit_and_fill(train_idx, [val_idx])
    final = fit_and_fill(IDX_DEV, [IDX_TEST, IDX_LIVE])
    if store:
        register(model, fset, out["P"], note=f"Dixon-Coles rho = {final['rho']:+.3f}")
        row = {"model": model, "feature_set": fset}
        for block, mask in (("cv", MASK_CV), ("test", MASK_TEST)):
            row.update({f"{block}_{k}": v for k, v in goal_metrics(out["lam_h"][mask], out["lam_a"][mask], mask).items()})
            total, both = (GOALS_H[mask] + GOALS_A[mask]) >= 3, (GOALS_H[mask] >= 1) & (GOALS_A[mask] >= 1)
            row[f"{block}_brier_over25"] = float(((out["over25"][mask] - total) ** 2).mean())
            row[f"{block}_brier_btts"] = float(((out["btts"][mask] - both) ** 2).mean())
        REG_RESULTS.append(row)
        GOALS[(model, fset)], FITTED[(model, fset)], GOAL_FIT_FNS[(model, fset)] = out, final, fit_fn
    return out


def run_linear_goal_difference(fset: str) -> None:
    """Plain linear regression on goal difference, scored by sign accuracy."""
    X, target = DATA[FEATURE_SETS[fset]], GOALS_H - GOALS_A
    pred = np.full(len(DATA), np.nan)
    for _, train_idx, val_idx in walk_forward_splits(CV_VAL_SEASONS):
        pred[val_idx] = make_pipeline(LinearRegression(), scale=True).fit(X.iloc[train_idx], target[train_idx]).predict(X.iloc[val_idx])
    final = make_pipeline(LinearRegression(), scale=True).fit(X.iloc[IDX_DEV], target[IDX_DEV])
    for idx in (IDX_TEST, IDX_LIVE):
        if len(idx):
            pred[idx] = final.predict(X.iloc[idx])
    row = {"model": "Linear regression (goal difference)", "feature_set": fset}
    for block, mask in (("cv", MASK_CV), ("test", MASK_TEST)):
        decisive = target[mask] != 0
        row[f"{block}_gd_mae"] = mean_absolute_error(target[mask], pred[mask])
        row[f"{block}_gd_rmse"] = float(np.sqrt(mean_squared_error(target[mask], pred[mask])))
        row[f"{block}_gd_sign_acc"] = float((np.sign(pred[mask]) == np.sign(target[mask])).mean())
        row[f"{block}_gd_sign_acc_decisive"] = float((np.sign(pred[mask][decisive]) == np.sign(target[mask][decisive])).mean())
    REG_RESULTS.append(row)
    # In the classification table it appears as a hard Home/Away call, so only accuracy-type metrics apply.
    hard = np.full((len(DATA), 3), np.nan)
    known = ~np.isnan(pred)
    hard[known] = np.column_stack([(pred[known] < 0), np.zeros(known.sum()), (pred[known] >= 0)]).astype(float)
    register("Linear regression (goal difference sign)", fset, hard, hard=True, note="hard Home/Away call, never predicts a draw")
    FITTED[("Linear regression (goal difference)", fset)] = final


REG_SHOW = ["model", "feature_set", "cv_mae_home", "cv_mae_away", "cv_rmse_home", "cv_rmse_away", "cv_poisson_dev_home",
            "cv_poisson_dev_away", "cv_gd_sign_acc", "test_mae_home", "test_mae_away", "test_rmse_home", "test_rmse_away",
            "test_poisson_dev_home", "test_poisson_dev_away", "test_gd_rmse", "test_gd_sign_acc", "test_gd_sign_acc_decisive"]


def show_regression(title: str) -> None:
    table = pd.DataFrame(REG_RESULTS)
    show(table[[c for c in REG_SHOW if c in table.columns]], title)


for _fset in FEATURE_SETS:
    run_linear_goal_difference(_fset)
show_regression("Goals regression after the linear baseline")
show_comparison("After the linear goal-difference baseline")

# %%
BEST_ALPHA: Dict[str, float] = {}
for _fset, _cols in FEATURE_SETS.items():
    _scores = {}
    for _alpha in CONFIG["poisson_alpha_grid"]:
        _dev = []
        for _, _tr, _va in walk_forward_splits(CV_VAL_SEASONS[-3:]):
            _bundle = make_poisson_glm_fit(_alpha)(DATA[_cols].iloc[_tr], GOALS_H[_tr], GOALS_A[_tr], SEASON[_tr])
            _lh, _la = goals_predict(_bundle, DATA[_cols].iloc[_va])
            _dev.append(mean_poisson_deviance(GOALS_H[_va], _lh) + mean_poisson_deviance(GOALS_A[_va], _la))
        _scores[_alpha] = float(np.mean(_dev))
    BEST_ALPHA[_fset] = min(_scores, key=_scores.get)
    print(f"{_fset}: Poisson GLM alpha grid (summed deviance) { {k: round(v, 4) for k, v in _scores.items()} } -> alpha = {BEST_ALPHA[_fset]}")
    run_goals_model("Poisson GLM + Dixon-Coles grid", _fset, make_poisson_glm_fit(BEST_ALPHA[_fset]))
show_regression("Goals regression after the Poisson GLM")
show_comparison("After the Poisson GLM")

# %%
for _fset in FEATURE_SETS:
    run_goals_model("XGBoost Poisson + Dixon-Coles grid", _fset, fit_xgb_poisson)
show_regression("Goals regression after XGBoost Poisson")
show_comparison("After XGBoost Poisson (all models)")

# %% [markdown]
# ## 7. Evaluation
#
# Everything below reads the stored predictions. No model is refitted and nothing is tuned on the holdout.
# The diagnostic cells are wrapped so that a failure in one plot cannot stop the rest of the notebook.
#
# ### 7.1 Final comparison table and chart
#
# One table with every model on every feature set: walk-forward CV, holdout and live blocks. The full table
# (including per-class precision and recall) is saved to `model_comparison.csv`.

# %%
def run_section(name: str, fn: Callable) -> object:
    """Run a diagnostic section. On error, print the traceback and carry on."""
    try:
        return fn()
    except Exception:
        print(f"[{name}] skipped after an error:")
        traceback.print_exc()
        return None


BOOKMAKER_KEY = ("Bookmaker (closing odds)", "-")
PRIOR_KEY = ("Class prior (majority class)", "-")
final_table = comparison_table()
final_table.to_csv(Path(CONFIG["out_dir"]) / "model_comparison.csv", index=False)
regression_table = pd.DataFrame(REG_RESULTS)
regression_table.to_csv(Path(CONFIG["out_dir"]) / "goals_regression_comparison.csv", index=False)

_candidates = final_table[final_table["cv_logloss"].notna() & (final_table["feature_set"] != "-")]
BEST_KEY = tuple(_candidates.iloc[0][["model", "feature_set"]])
_no_odds = _candidates[_candidates["feature_set"] != "C_with_market"]
BEST_NO_ODDS_KEY = tuple(_no_odds.iloc[0][["model", "feature_set"]])
BEST_XGB_FSET = min(FEATURE_SETS, key=lambda f: RESULTS[("XGBoost", f)]["cv_logloss"])
BEST_GOALS_KEY = min(GOALS, key=lambda k: RESULTS[k]["cv_logloss"])
print(f"Best model by walk-forward log loss: {BEST_KEY}")
print(f"Best model without odds features:    {BEST_NO_ODDS_KEY}")
print(f"Best XGBoost feature set:            {BEST_XGB_FSET}")
print(f"Best goals model:                    {BEST_GOALS_KEY}")

show_comparison("Final model comparison")
show(final_table[["model", "feature_set", "test_acc", "test_macro_f1", "test_prec_home", "test_rec_home",
                  "test_prec_draw", "test_rec_draw", "test_prec_away", "test_rec_away"]],
     "Holdout precision and recall by class (draws are the hard class)")
show_regression("Goals regression comparison")


def plot_model_comparison(table: pd.DataFrame) -> None:
    """Log loss of every probabilistic model: walk-forward CV and holdout, coloured by feature set."""
    prob = table[table["cv_logloss"].notna() & (table["feature_set"] != "-")]
    models = list(dict.fromkeys(prob.sort_values("cv_logloss")["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.5 * len(models) + 2.6), sharey=True)
    for ax, col, title in zip(axes, ("cv_logloss", "test_logloss"),
                              ("Walk-forward CV log loss", "Holdout log loss")):
        for k, fset in enumerate(FEATURE_SETS):
            sub = prob[prob["feature_set"] == fset].set_index("model").reindex(models)
            ax.scatter(sub[col], range(len(models)), s=80, color=SERIES[k], label=fset, zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
        for key, label, style in ((BOOKMAKER_KEY, "bookmaker", "--"), (PRIOR_KEY, "class prior", ":")):
            value = RESULTS[key].get(col, np.nan)
            if np.isfinite(value):
                ax.axvline(value, color=INK_SOFT, linestyle=style, linewidth=1.2, zorder=1)
                ax.text(value, -0.85, label, ha="center", va="bottom", color=INK_SOFT, fontsize=9)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_ylim(len(models) - 0.5, -1.4)
        ax.set_xlabel("Log loss (lower is better)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[1].legend(title="Feature set", loc="lower right")
    fig.tight_layout()
    plt.show()


run_section("comparison chart", lambda: plot_model_comparison(final_table))

# %% [markdown]
# ### 7.2 Confusion matrix
#
# Rows are actual results and columns are predictions on the holdout. The draw column is usually almost empty:
# a draw is rarely the single most likely outcome, so a model can be well calibrated on draws and still
# almost never predict one.

# %%
def plot_confusion(keys: List[Tuple[str, str]]) -> None:
    """Holdout confusion matrices (counts, shaded by row share)."""
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.6))
    for ax, key in zip(np.atleast_1d(axes), keys):
        matrix = confusion_matrix(Y[MASK_TEST], PREDS[key][MASK_TEST].argmax(axis=1), labels=[2, 1, 0])
        share = matrix / matrix.sum(axis=1, keepdims=True)
        sns.heatmap(share, annot=matrix, fmt="d", cmap=SEQ_CMAP, vmin=0, vmax=1, cbar=False, linewidths=2,
                    linecolor=SURFACE, ax=ax, xticklabels=["Home", "Draw", "Away"], yticklabels=["Home", "Draw", "Away"])
        ax.set_title(f"{key[0]}\n{key[1]}" if key[1] != "-" else key[0], fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
    fig.suptitle("Holdout confusion matrices (cell = matches, shade = share of actual class)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    plt.show()


run_section("confusion matrix", lambda: plot_confusion(list(dict.fromkeys([BEST_KEY, BEST_NO_ODDS_KEY, BOOKMAKER_KEY]))))

# %% [markdown]
# ### 7.3 Reliability diagrams before and after calibration
#
# For each class the holdout matches are grouped into equal-sized bins of predicted probability. A perfectly
# calibrated model sits on the diagonal.

# %%
def reliability_points(p: np.ndarray, hit: np.ndarray, n_bins: int = 8) -> pd.DataFrame:
    """Mean predicted probability and observed frequency per quantile bin."""
    bins = pd.qcut(p, n_bins, duplicates="drop")
    return pd.DataFrame({"p": p, "hit": hit}).groupby(bins, observed=True).mean()


def plot_reliability(fset: str) -> None:
    """Reliability curves on the holdout for logistic regression and XGBoost, raw and calibrated."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for r, model in enumerate(("Logistic regression", "XGBoost")):
        for c, k in enumerate((2, 1, 0)):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], color=INK_SOFT, linestyle=":", linewidth=1)
            for label, key, color in (("before calibration", (model, fset), PALETTE["blue"]),
                                      ("after calibration", (f"{model} (calibrated)", fset), PALETTE["orange"])):
                points = reliability_points(PREDS[key][MASK_TEST][:, k], (Y[MASK_TEST] == k).astype(float))
                ax.plot(points["p"], points["hit"], marker="o", color=color, label=label)
            ax.set_title(f"{model}: {CLASS_NAMES[k]}", fontsize=10)
            ax.set_xlim(0, 0.9)
            ax.set_ylim(0, 0.9)
            if r == 1:
                ax.set_xlabel("Predicted probability")
            if c == 0:
                ax.set_ylabel("Observed frequency")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"Reliability on the holdout, feature set {fset}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    plt.show()


run_section("reliability diagrams", lambda: plot_reliability(BEST_KEY[1]))

# %% [markdown]
# ### 7.4 Performance by season
#
# Log loss per season for the benchmark and the selected models. Walk-forward seasons come first, then the
# holdout, then the live block. The live season has few matches, so its point is very noisy.

# %%
def plot_per_season(keys: List[Tuple[str, str]]) -> None:
    """Per-season log loss across walk-forward, holdout and live seasons."""
    seasons = CV_VAL_SEASONS + TEST_SEASONS + LIVE_SEASONS
    styles = {BOOKMAKER_KEY: (INK, "--"), PRIOR_KEY: (INK_SOFT, ":")}
    fig, ax = plt.subplots(figsize=(12, 5))
    colour_index = 0
    for key in keys:
        values = []
        for season in seasons:
            mask = SEASON == season
            P = PREDS[key][mask]
            values.append(np.nan if (mask.sum() == 0 or np.isnan(P).any()) else per_match_log_loss(P, Y[mask]).mean())
        if key in styles:
            colour, line = styles[key]
        else:
            colour, line = SERIES[colour_index], "-"
            colour_index += 1
        label = key[0] if key[1] == "-" else f"{key[0]} [{key[1]}]"
        ax.plot(range(len(seasons)), values, marker="o", color=colour, linestyle=line, label=label)
    first_test = len(CV_VAL_SEASONS)
    ax.axvspan(first_test - 0.5, first_test + len(TEST_SEASONS) - 0.5, color=GRID, alpha=0.6, zorder=0)
    ax.text(first_test - 0.4, ax.get_ylim()[1], "holdout", va="top", color=INK_SOFT)
    labels = [season_label(s) + (f"\n(live, n={int((SEASON == s).sum())})" if s in LIVE_SEASONS else "") for s in seasons]
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Log loss by season")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    plt.show()


run_section("per-season plot", lambda: plot_per_season(list(dict.fromkeys([BOOKMAKER_KEY, BEST_KEY, BEST_NO_ODDS_KEY, PRIOR_KEY]))))

# %% [markdown]
# ### 7.5 Model versus bookmaker
#
# The difference in mean log loss and RPS between each selected model and the bookmaker benchmark on the
# holdout, with a paired bootstrap confidence interval (matches are resampled with replacement and both
# forecasters are scored on the same resample). A negative difference means the model beats the bookmaker.
# An interval that includes zero means the data cannot separate the two.

# %%
def paired_bootstrap(P_model: np.ndarray, P_ref: np.ndarray, y: np.ndarray, n_boot: int) -> dict:
    """Bootstrap the mean difference (model minus reference) in log loss and RPS."""
    rng = np.random.default_rng(CONFIG["seed"])
    diffs = {"logloss": per_match_log_loss(P_model, y) - per_match_log_loss(P_ref, y),
             "rps": per_match_rps(P_model, y) - per_match_rps(P_ref, y)}
    sample = rng.integers(0, len(y), size=(n_boot, len(y)))
    out = {}
    for name, diff in diffs.items():
        means = diff[sample].mean(axis=1)
        out[f"{name}_diff"] = float(diff.mean())
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    return out


def bookmaker_comparison() -> pd.DataFrame:
    keys = list(dict.fromkeys([BEST_KEY, BEST_NO_ODDS_KEY, ("XGBoost (calibrated)", BEST_XGB_FSET),
                               ("Logistic regression (calibrated)", "B_form_plus_elo"), BEST_GOALS_KEY]))
    rows = []
    for key in keys:
        result = paired_bootstrap(PREDS[key][MASK_TEST], PREDS[BOOKMAKER_KEY][MASK_TEST], Y[MASK_TEST], CONFIG["n_bootstrap"])
        verdict = "better" if result["logloss_ci_high"] < 0 else ("worse" if result["logloss_ci_low"] > 0 else "not separable")
        rows.append({"model": key[0], "feature_set": key[1], **result, "vs bookmaker (log loss)": verdict})
    table = pd.DataFrame(rows)
    show(table, f"Model minus bookmaker on the holdout ({int(MASK_TEST.sum())} matches, "
                f"{CONFIG['n_bootstrap']} paired bootstrap resamples, 95% interval)")
    return table


bookmaker_table = run_section("bookmaker comparison", bookmaker_comparison)

# %% [markdown]
# ### 7.6 Feature importance and SHAP
#
# Three views of the best XGBoost model: gain (how much each feature improves the training objective),
# permutation importance on the holdout (how much holdout log loss worsens when a feature is shuffled) and SHAP
# values on a sample of 2,000 matches. Correlated features share importance between them, so read groups of
# related features together rather than individual ranks.

# %%
TOP_FEATURES = pd.DataFrame()


def feature_importance(fset: str, plot: bool = True) -> pd.DataFrame:
    """Gain and holdout permutation importance for the XGBoost model on one feature set."""
    cols = FEATURE_SETS[fset]
    pipeline = FITTED[("XGBoost", fset)]
    booster = pipeline.named_steps["model"].get_booster()
    gain = pd.Series(0.0, index=cols)
    for name, value in booster.get_score(importance_type="gain").items():
        gain[cols[int(name[1:])] if name[1:].isdigit() and name.startswith("f") else name] = value
    perm = permutation_importance(pipeline, DATA[cols].iloc[IDX_TEST], Y[IDX_TEST], scoring="neg_log_loss",
                                  n_repeats=3, random_state=CONFIG["seed"], n_jobs=1)
    table = pd.DataFrame({"gain": gain / gain.sum(), "permutation_logloss_increase": perm.importances_mean}, index=cols)
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
        for ax, col, title, colour in ((axes[0], "gain", "XGBoost gain (share of total)", PALETTE["blue"]),
                                       (axes[1], "permutation_logloss_increase", "Permutation importance on the holdout\n(increase in log loss)", PALETTE["orange"])):
            top = table[col].sort_values(ascending=False).head(15)[::-1]
            ax.barh(top.index, top.values, color=colour, height=0.65)
            ax.set_title(title)
            ax.grid(axis="y", visible=False)
        fig.suptitle(f"Feature importance: XGBoost on {fset}", x=0.02, ha="left", fontweight="bold")
        fig.tight_layout()
        plt.show()
    return table


def importance_section() -> pd.DataFrame:
    table = feature_importance(BEST_XGB_FSET)
    top = pd.DataFrame({
        "rank": range(1, 11),
        "by gain": table["gain"].sort_values(ascending=False).head(10).index,
        "gain share": table["gain"].sort_values(ascending=False).head(10).values,
        "by permutation": table["permutation_logloss_increase"].sort_values(ascending=False).head(10).index,
        "log loss increase": table["permutation_logloss_increase"].sort_values(ascending=False).head(10).values})
    show(top, f"Top 10 features, XGBoost on {BEST_XGB_FSET}")
    if BEST_XGB_FSET == "C_with_market":      # also show what matters once the odds are removed
        no_odds = feature_importance("B_form_plus_elo", plot=False)
        show(no_odds.sort_values("gain", ascending=False).head(10), "Top 10 by gain without odds features (XGBoost on B_form_plus_elo)")
    return top


TOP_FEATURES = run_section("feature importance", importance_section)


def shap_section() -> None:
    """SHAP values from XGBoost's exact tree contributions, plotted with the shap package."""
    cols = FEATURE_SETS[BEST_XGB_FSET]
    pipeline = FITTED[("XGBoost", BEST_XGB_FSET)]
    pool = np.concatenate([IDX_DEV, IDX_TEST])
    sample = np.sort(np.random.default_rng(CONFIG["seed"]).choice(pool, size=min(CONFIG["shap_sample"], len(pool)), replace=False))
    X_sample = pipeline[:-1].transform(DATA[cols].iloc[sample])
    contrib = pipeline.named_steps["model"].get_booster().predict(xgb.DMatrix(X_sample), pred_contribs=True)
    contrib = np.asarray(contrib).reshape(len(sample), 3, len(cols) + 1)[:, :, :-1]      # drop the bias term
    mean_abs = pd.Series(np.abs(contrib).mean(axis=(0, 1)), index=cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top = mean_abs.head(15)[::-1]
    ax.barh(top.index, top.values, color=PALETTE["blue"], height=0.65)
    ax.set_title(f"Mean |SHAP| across the three classes (XGBoost, {BEST_XGB_FSET})")
    ax.set_xlabel("Mean absolute contribution to the log-odds")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    plt.show()
    try:
        import shap
        shap.summary_plot(contrib[:, 2, :], pd.DataFrame(X_sample, columns=cols), max_display=15, show=False)
        plt.title("SHAP summary for the Home-win class", loc="left", fontweight="bold")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"SHAP beeswarm plot unavailable ({type(exc).__name__}: {exc}); the bar chart above uses the same values.")


run_section("SHAP", shap_section)

# %% [markdown]
# ### 7.7 Value-bet backtest (illustrative only)
#
# Bet one unit on any outcome where `model probability x odds > 1.05`, on the holdout seasons. Two prices are
# shown: the pre-match market average and the best pre-match price across bookmakers.
#
# **Warning.** This is an illustration, not a strategy. Past ROI does not imply future profit. The result is
# very sensitive to the odds source and timing: the average price is rarely available in full, the best price
# requires accounts everywhere, the quoted odds are snapshots that move before kickoff, and bookmakers limit
# winning accounts. With a few hundred bets the bootstrap interval is wide, and an interval that includes zero
# means there is no evidence of an edge. A model that takes the market's own probabilities as inputs will also
# rarely disagree with the market enough to bet.

# %%
def value_bet_backtest(key: Tuple[str, str], odds_kind: str) -> dict:
    """Flat one-unit stakes whenever the model's expected return exceeds the threshold."""
    odds = DATA.loc[MASK_TEST, [f"{odds_kind}_odds_away", f"{odds_kind}_odds_draw", f"{odds_kind}_odds_home"]].to_numpy(dtype=float)
    P = PREDS[key][MASK_TEST]
    bets = (P * odds > CONFIG["value_bet_threshold"]) & ~np.isnan(odds)
    won = np.eye(3, dtype=bool)[Y[MASK_TEST]]
    profit = np.where(won, odds - 1.0, -1.0)[bets]
    row = {"model": key[0], "feature_set": key[1], "odds": "market average" if odds_kind == "open" else "best price",
           "bets": int(bets.sum()), "hit_rate": np.nan, "roi": np.nan, "roi_ci_low": np.nan, "roi_ci_high": np.nan,
           "profit_units": float(profit.sum())}
    if len(profit) >= 10:
        rng = np.random.default_rng(CONFIG["seed"])
        boot = profit[rng.integers(0, len(profit), size=(CONFIG["n_bootstrap"], len(profit)))].mean(axis=1)
        row.update({"hit_rate": float((profit > 0).mean()), "roi": float(profit.mean()),
                    "roi_ci_low": float(np.percentile(boot, 2.5)), "roi_ci_high": float(np.percentile(boot, 97.5))})
    return row


def backtest_section() -> pd.DataFrame:
    keys = list(dict.fromkeys([BEST_KEY, BEST_NO_ODDS_KEY]))
    table = pd.DataFrame([value_bet_backtest(key, kind) for key in keys for kind in ("open", "best")])
    show(table, f"Illustrative value-bet backtest on the holdout (threshold {CONFIG['value_bet_threshold']}, "
                f"{int(MASK_TEST.sum())} matches, 95% bootstrap interval on ROI)")
    return table


backtest_table = run_section("value-bet backtest", backtest_section)

# %% [markdown]
# ### 7.8 xG-enabled experiment (2014/15 onward)
#
# Understat xG only exists from 2014/15, so it is tested in a separate experiment instead of shrinking the main
# training set. Both arms are trained on 2014/15 onward only and scored on the same holdout: one arm uses
# `B_form_plus_elo`, the other adds the rolling xG block. The comparison isolates what xG adds. The absolute
# numbers are not comparable to the main table because the training window is much shorter.

# %%
def xg_experiment() -> pd.DataFrame:
    if not HAS_XG:
        print("xG data did not load: experiment skipped (see the Data availability log).")
        return pd.DataFrame()
    start = CONFIG["xg_first_season"]
    val_seasons = [s for s in CV_SCORED if s >= start + 4]
    base_cols = FEATURE_SETS["B_form_plus_elo"]
    arms = {"B_form_plus_elo (2014/15+)": base_cols, "D_form_elo_xg (2014/15+)": base_cols + XG_FEATURES}
    fits = {"Logistic regression": make_logreg_fit(BEST_C["B_form_plus_elo"]),
            "XGBoost": make_xgb_fit(XGB_PARAMS["B_form_plus_elo"])}
    rows = []
    cv_mask = np.isin(SEASON, val_seasons)
    for arm, cols in arms.items():
        for model, fit_fn in fits.items():
            P = run_classifier(model, arm, fit_fn, cols=cols, val_seasons=val_seasons, min_season=start, store=False)
            cv, test = prob_metrics(P[cv_mask], Y[cv_mask]), prob_metrics(P[MASK_TEST], Y[MASK_TEST])
            rows.append({"model": model, "feature_set": arm, "n_features": len(cols), "cv_logloss": cv["logloss"],
                         "cv_rps": cv["rps"], "test_logloss": test["logloss"], "test_rps": test["rps"],
                         "test_brier": test["brier"], "test_acc": test["acc"]})
    table = pd.DataFrame(rows)
    table.to_csv(Path(CONFIG["out_dir"]) / "xg_experiment.csv", index=False)
    show(table, f"xG experiment (walk-forward seasons {season_label(val_seasons[0])} to {season_label(val_seasons[-1])}, same holdout)")
    return table


xg_table = run_section("xG experiment", xg_experiment)

# %% [markdown]
# ## 8. Predicting upcoming fixtures
#
# Fixtures come from the football-data.co.uk `fixtures.csv`. If that file cannot be reached, fill in
# `MANUAL_FIXTURES` below (team names as used by football-data.co.uk, odds columns optional) and re-run this
# section.
#
# The fixtures are appended to the results table as rows without a score and passed through the same
# `build_features` function, so they can only see matches played before them. The best classifier and the best
# goals model are refitted on every completed match (development, holdout and live) before predicting. Their
# hyperparameters, boosting rounds, calibrators and ensemble weight stay exactly as chosen on the development
# seasons.

# %%
# Edit this only if fixtures.csv is unreachable. Example:
# MANUAL_FIXTURES = pd.DataFrame([
#     {"Date": "17/10/2026", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "AvgH": 1.80, "AvgD": 3.90, "AvgA": 4.40},
# ])
MANUAL_FIXTURES = pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def load_upcoming_fixtures(matches: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Next gameweek of EPL fixtures, shaped like the results table but with no scores."""
    source, raw = "football-data.co.uk fixtures.csv", None
    try:
        response = SESSION.get(FIXTURES_URL, timeout=30)
        response.raise_for_status()
        raw = parse_football_data_csv(response.content)
        raw = raw[raw["Div"].astype(str).str.strip() == "E0"]
    except Exception as exc:
        print(f"fixtures.csv unavailable ({type(exc).__name__}): falling back to MANUAL_FIXTURES")
    if raw is None or raw.empty:
        raw, source = MANUAL_FIXTURES.copy(), "MANUAL_FIXTURES"
    if raw.empty:
        return pd.DataFrame(), source
    fixtures = harmonise_schema(raw)
    fixtures["Date"] = parse_dates(fixtures["Date"])
    for col in NUMERIC_COLS:
        fixtures[col] = pd.to_numeric(fixtures[col], errors="coerce")
    fixtures["HomeTeam"], fixtures["AwayTeam"] = fixtures["HomeTeam"].map(standardise_team), fixtures["AwayTeam"].map(standardise_team)
    fixtures = fixtures.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    fixtures = fixtures[fixtures["Date"] > matches["Date"].max()].sort_values(["Date", "HomeTeam"])
    if fixtures.empty:
        return pd.DataFrame(), source
    fixtures = fixtures[fixtures["Date"] <= fixtures["Date"].min() + pd.Timedelta(days=6)]      # one gameweek
    fixtures = fixtures.drop_duplicates(["HomeTeam", "AwayTeam"])
    known = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    unknown = sorted((set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])) - known)
    if unknown:
        print(f"Warning: clubs with no history in the data (check spelling against TEAM_NAME_MAP): {unknown}")
    fixtures = fixtures.reindex(columns=matches.columns)
    fixtures[BLANK_ON_FIXTURE] = np.nan
    fixtures["season_start"] = season_start_from_date(fixtures["Date"])
    fixtures["season"] = fixtures["season_start"].map(season_label)
    fixtures["is_fixture"] = True
    fixtures["match_id"] = np.arange(len(fixtures)) + int(matches["match_id"].max()) + 1
    return fixtures.reset_index(drop=True), source


def refit_function(key: Tuple[str, str]) -> Callable:
    """How to refit a base model on more data without re-running early stopping on a partial season."""
    model, fset = key
    if model == "XGBoost":
        return make_xgb_fit(XGB_PARAMS[fset], rounds=FITTED[key].named_steps["model"].n_estimators)
    if key in GOAL_FIT_FNS and "XGBoost" in model:
        return lambda X, gh, ga, s: fit_xgb_poisson(X, gh, ga, s, rounds=FITTED[key]["rounds"])
    return GOAL_FIT_FNS[key] if key in GOAL_FIT_FNS else FIT_FNS[key]


def build_goals_bundle(key: Tuple[str, str], idx: np.ndarray) -> dict:
    """Refit a goals model on the given rows and re-estimate the Dixon-Coles rho."""
    X = DATA[FEATURE_SETS[key[1]]].iloc[idx]
    bundle = refit_function(key)(X, GOALS_H[idx], GOALS_A[idx], SEASON[idx])
    fit_h, fit_a = goals_predict(bundle, X)
    bundle["rho"] = fit_rho(GOALS_H[idx], GOALS_A[idx], fit_h, fit_a)
    return {"kind": "goals", "name": key[0], "feature_set": key[1], "features": FEATURE_SETS[key[1]], "goals": bundle}


def build_classifier_bundle(key: Tuple[str, str], idx: np.ndarray) -> dict:
    """Refit the chosen 1X2 model (plain, calibrated, ensemble or goals-based) on the given rows."""
    model, fset = key
    if key in GOALS:
        return build_goals_bundle(key, idx)
    X = DATA[FEATURE_SETS[fset]].iloc[idx]
    bundle = {"kind": "classifier", "name": model, "feature_set": fset, "features": FEATURE_SETS[fset],
              "components": [], "class_names": CLASS_NAMES}
    if model.startswith("Ensemble"):
        weight = ENSEMBLE_WEIGHTS[fset]
        parts = [("Logistic regression", weight), ("XGBoost", 1.0 - weight)]
    else:
        parts = [(model.replace(" (calibrated)", ""), 1.0)]
    for base, weight in parts:
        calibrated = model.startswith("Ensemble") or model.endswith("(calibrated)")
        bundle["components"].append({
            "name": base, "weight": weight, "model": refit_function((base, fset))(X, Y[idx], SEASON[idx]),
            "calibrator": CALIBRATORS[(f"{base} (calibrated)", fset)] if calibrated else None})
    return bundle


def bundle_predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Probabilities [Away, Draw, Home] from a saved bundle."""
    X = X[bundle["features"]]
    if bundle["kind"] == "goals":
        lam_h, lam_a = goals_predict(bundle["goals"], X)
        return grid_summary(scoreline_grid(lam_h, lam_a, bundle["goals"]["rho"], CONFIG["max_goals"]))["P"]
    P = np.zeros((len(X), 3))
    for part in bundle["components"]:
        p = part["model"].predict_proba(X)
        if part["calibrator"] is not None:
            p = apply_calibrator(part["calibrator"], p)
        P += part["weight"] * p
    return P / P.sum(axis=1, keepdims=True)


# Consistency check: a bundle rebuilt on the development rows must reproduce the stored holdout predictions.
_check = build_classifier_bundle(BEST_KEY, IDX_DEV)
_gap = float(np.abs(bundle_predict_proba(_check, DATA.iloc[IDX_TEST]) - PREDS[BEST_KEY][IDX_TEST]).max())
print(f"Bundle check for {BEST_KEY}: max probability gap versus stored holdout predictions = {_gap:.4f}")

IDX_ALL = np.arange(len(DATA))
FINAL_CLASSIFIER = build_classifier_bundle(BEST_KEY, IDX_ALL)
FINAL_GOALS = build_goals_bundle(BEST_GOALS_KEY, IDX_ALL)
print(f"Final models refitted on {len(IDX_ALL)} matches through {DATA['Date'].max().date()}")

upcoming, fixtures_source = load_upcoming_fixtures(matches)
PRED_COLS = ["Date", "HomeTeam", "AwayTeam", "P(H)", "P(D)", "P(A)", "predicted_scoreline", "xG_home", "xG_away",
             "P(over 2.5)", "P(BTTS)", "market_P(H)", "market_P(D)", "market_P(A)"]
if upcoming.empty:
    predictions = pd.DataFrame(columns=PRED_COLS)
    print("No upcoming fixtures found. Fill in MANUAL_FIXTURES above and re-run this section.")
else:
    combined = pd.concat([matches, upcoming], ignore_index=True)
    combined["is_fixture"] = combined["is_fixture"].astype(bool)
    if HAS_CLUBELO:
        combined = safe_optional(load_clubelo, combined, "ClubElo (fixtures)", ["clubelo_home", "clubelo_away"])
    fixture_features, _ = build_features(combined, CONFIG)
    X_next = fixture_features[fixture_features["is_fixture"]].reset_index(drop=True)
    # Only data available before the first fixture is used.
    assert combined.loc[~combined["is_fixture"], "Date"].max() < X_next["Date"].min(), "results must precede fixtures"
    assert DATA["Date"].max() < X_next["Date"].min()
    P_next = bundle_predict_proba(FINAL_CLASSIFIER, X_next)
    assert np.allclose(P_next.sum(axis=1), 1.0, atol=1e-6)
    lam_h, lam_a = goals_predict(FINAL_GOALS["goals"], X_next[FINAL_GOALS["features"]])
    summary = grid_summary(scoreline_grid(lam_h, lam_a, FINAL_GOALS["goals"]["rho"], CONFIG["max_goals"]))
    predictions = pd.DataFrame({
        "Date": X_next["Date"].dt.date, "HomeTeam": X_next["HomeTeam"], "AwayTeam": X_next["AwayTeam"],
        "P(H)": P_next[:, 2], "P(D)": P_next[:, 1], "P(A)": P_next[:, 0],
        "predicted_scoreline": [f"{h}-{a}" for h, a in zip(summary["mode_home"], summary["mode_away"])],
        "xG_home": lam_h, "xG_away": lam_a, "P(over 2.5)": summary["over25"], "P(BTTS)": summary["btts"],
        "market_P(H)": X_next["mkt_p_home"], "market_P(D)": X_next["mkt_p_draw"], "market_P(A)": X_next["mkt_p_away"]})
    if "market" in BEST_KEY[1] and X_next["mkt_p_home"].isna().any():
        print("Warning: some fixtures have no odds yet, so their market features were imputed with training medians.")
    print(f"Fixtures source: {fixtures_source} | 1X2 model: {BEST_KEY} | goals model: {BEST_GOALS_KEY}")
    print(f"Latest result in the data: {matches['Date'].max().date()} | fixtures from {X_next['Date'].min().date()} "
          f"to {X_next['Date'].max().date()} | run date: {date.today()}")
predictions.to_csv(Path(CONFIG["out_dir"]) / "predictions_next_gameweek.csv", index=False)
show(predictions.round(3), "Predictions for the next gameweek")

# %% [markdown]
# ### Predicted results for every remaining fixture
#
# The cell above covers one gameweek. This cell predicts a plain result for **every match still to be played
# this season**: who wins (or a draw) and a scoreline. No probabilities or odds are written to the output.
#
# - **Fixture list.** football-data.co.uk only lists the next round, so the full schedule comes from the public
#   Fantasy Premier League API. If it cannot be reached, the cell falls back to the next-gameweek fixtures.
# - **Form is frozen at today.** Nobody knows the results in between, so each fixture is predicted as if it were
#   the next match for both clubs, using their form, league position and Elo as of the latest result.
# - **Model.** Bookmaker odds only exist a few days ahead. Fixtures in the next-gameweek table keep that
#   table's prediction. All other fixtures use the best model that needs no odds (form and Elo only).
# - **Result and score.** The result is the most likely of home win, draw and away win. The score is the most
#   likely scoreline that agrees with that result.
#
# Treat anything beyond the next gameweek as a rough guide and re-run the notebook each week. Draws are rarely
# the single most likely outcome, so few or none will be predicted even though about a quarter of matches end
# level.

# %%
FPL_API = "https://fantasy.premierleague.com/api/"
RESULT_LABELS = {0: "Away win", 1: "Draw", 2: "Home win"}       # same order as CLASS_NAMES
SEASON_PRED_COLS = ["gameweek", "Date", "HomeTeam", "AwayTeam", "predicted_result", "predicted_winner",
                    "predicted_score", "based_on"]


def load_remaining_fixtures(matches: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Every unplayed fixture of the current season, shaped like the results table but with no scores."""
    season = int(matches["season_start"].max())
    current = matches[matches["season_start"] == season]
    played = set(zip(current["HomeTeam"], current["AwayTeam"]))
    last_result = matches["Date"].max()
    try:
        teams = SESSION.get(FPL_API + "bootstrap-static/", timeout=30)
        teams.raise_for_status()
        names = {t["id"]: standardise_team(t["name"]) for t in teams.json()["teams"]}
        listing = SESSION.get(FPL_API + "fixtures/", timeout=30)
        listing.raise_for_status()
        raw = pd.DataFrame([{
            "gameweek": f.get("event"), "kickoff": f.get("kickoff_time"), "finished": bool(f.get("finished")),
            "HomeTeam": names[f["team_h"]], "AwayTeam": names[f["team_a"]]} for f in listing.json()])
        kickoff = pd.to_datetime(raw["kickoff"], utc=True, errors="coerce")
        raw["Date"] = kickoff.dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
        source = "Fantasy Premier League API"
    except Exception as exc:
        print(f"FPL fixture list unavailable ({type(exc).__name__}): only the next gameweek will be predicted")
        raw = upcoming[["Date", "HomeTeam", "AwayTeam"]].copy() if len(upcoming) else pd.DataFrame(
            columns=["Date", "HomeTeam", "AwayTeam"])
        raw["gameweek"], raw["finished"] = np.nan, False
        source = fixtures_source
    if raw.empty:
        return pd.DataFrame(), source
    is_played = pd.Series([pair in played for pair in zip(raw["HomeTeam"], raw["AwayTeam"])], index=raw.index)
    raw = raw[~raw["finished"] & ~is_played].drop_duplicates(["HomeTeam", "AwayTeam"])
    known = set(current["HomeTeam"]) | set(current["AwayTeam"])
    unknown = sorted((set(raw["HomeTeam"]) | set(raw["AwayTeam"])) - known)
    if unknown:
        print(f"Warning: clubs not in this season's results (check TEAM_NAME_MAP), fixtures skipped: {unknown}")
        raw = raw[raw["HomeTeam"].isin(known) & raw["AwayTeam"].isin(known)]
    if raw.empty:
        return pd.DataFrame(), source
    # Postponed matches have no date yet. They get a placeholder after the last scheduled match so that
    # the feature code can sort them; the output shows their date as blank.
    raw["date_known"] = raw["Date"].notna()
    placeholder = max(raw["Date"].max() if raw["date_known"].any() else last_result, last_result) + pd.Timedelta(days=1)
    raw["Date"] = raw["Date"].fillna(placeholder)
    raw.loc[raw["Date"] <= last_result, "Date"] = last_result + pd.Timedelta(days=1)   # must follow every result
    raw = raw.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    fixtures = raw.reindex(columns=matches.columns)
    fixtures[BLANK_ON_FIXTURE] = np.nan
    fixtures["season_start"] = season
    fixtures["season"] = season_label(season)
    fixtures["is_fixture"] = True
    fixtures["match_id"] = np.arange(len(fixtures)) + int(matches["match_id"].max()) + 1_000
    fixtures["gameweek"], fixtures["date_known"] = raw["gameweek"].values, raw["date_known"].values
    # Round k holds each club's k-th remaining match, so no club appears twice in one round.
    seen: Dict[str, int] = defaultdict(int)
    rounds = []
    for home, away in zip(fixtures["HomeTeam"], fixtures["AwayTeam"]):
        k = max(seen[home], seen[away])
        seen[home], seen[away] = k + 1, k + 1
        rounds.append(k)
    fixtures["round"] = rounds
    return fixtures, source


def remaining_fixture_features(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Pre-match features for every remaining fixture, with each club's form frozen at the latest result.

    Each round is appended to the results on its own, so `build_features` treats every fixture as the next
    match for both clubs and never sees another unplayed fixture in a rolling window.
    """
    extra = ["gameweek", "date_known", "round"]
    combined = pd.concat([matches, fixtures.drop(columns=extra)], ignore_index=True)
    combined["is_fixture"] = combined["is_fixture"].astype(bool)
    if HAS_CLUBELO:
        combined = safe_optional(load_clubelo, combined, "ClubElo (remaining fixtures)", ["clubelo_home", "clubelo_away"])
    results_part, fixtures_part = combined[~combined["is_fixture"]], combined[combined["is_fixture"]]
    round_of = fixtures.set_index("match_id")["round"]
    blocks = []
    for k in sorted(round_of.unique()):
        block = fixtures_part[fixtures_part["match_id"].map(round_of) == k]
        table, _ = build_features(pd.concat([results_part, block], ignore_index=True), CONFIG)
        blocks.append(table[table["is_fixture"]])
    X = pd.concat(blocks, ignore_index=True)
    return X.merge(fixtures[["match_id"] + extra], on="match_id", how="left")


def result_and_score(P: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Most likely result per match and the most likely scoreline that agrees with it."""
    result = P.argmax(axis=1)                                   # 0 = away win, 1 = draw, 2 = home win
    n_goals = grid.shape[1]
    i, j = np.indices((n_goals, n_goals))
    region = {0: i < j, 1: i == j, 2: i > j}
    scores = []
    for row, outcome in enumerate(result):
        flat = np.where(region[int(outcome)], grid[row], -1.0).argmax()
        scores.append(f"{flat // n_goals}-{flat % n_goals}")
    return result, scores


def predict_remaining_season() -> pd.DataFrame:
    fixtures, source = load_remaining_fixtures(matches)
    if fixtures.empty:
        print("No remaining fixtures found for this season.")
        return pd.DataFrame(columns=SEASON_PRED_COLS)
    _t = time.time()
    X = remaining_fixture_features(matches, fixtures)
    assert DATA["Date"].max() < X["Date"].min(), "results must precede fixtures"
    # Models that need no odds, refitted on every completed match.
    goals_key = min((k for k in GOALS if k[1] != "C_with_market"), key=lambda k: RESULTS[k]["cv_logloss"])
    classifier = build_classifier_bundle(BEST_NO_ODDS_KEY, IDX_ALL)
    goals = build_goals_bundle(goals_key, IDX_ALL)
    P = bundle_predict_proba(classifier, X)
    lam_h, lam_a = goals_predict(goals["goals"], X[goals["features"]])
    grid = scoreline_grid(lam_h, lam_a, goals["goals"]["rho"], CONFIG["max_goals"])
    result, scores = result_and_score(P, grid)
    table = pd.DataFrame({
        "gameweek": pd.to_numeric(X["gameweek"], errors="coerce").astype("Int64"),
        "Date": X["Date"].dt.date.where(X["date_known"].astype(bool), None),
        "HomeTeam": X["HomeTeam"], "AwayTeam": X["AwayTeam"], "result": result, "predicted_score": scores,
        "based_on": "form and Elo", "sort_date": X["Date"]})
    # Fixtures already in the next-gameweek table keep that table's prediction, which also used the odds.
    if len(predictions):
        P_week = predictions[["P(A)", "P(D)", "P(H)"]].to_numpy(dtype=float)
        grid_week = scoreline_grid(predictions["xG_home"].to_numpy(dtype=float), predictions["xG_away"].to_numpy(dtype=float),
                                   FINAL_GOALS["goals"]["rho"], CONFIG["max_goals"])
        week_result, week_scores = result_and_score(P_week, grid_week)
        has_odds = predictions["market_P(H)"].notna().to_numpy() if "market" in BEST_KEY[1] else np.ones(len(predictions), bool)
        week = {(h, a): (r, s, "form, Elo and market odds" if odds and "market" in BEST_KEY[1] else "form and Elo")
                for h, a, r, s, odds in zip(predictions["HomeTeam"], predictions["AwayTeam"], week_result, week_scores, has_odds)}
        for idx, pair in enumerate(zip(table["HomeTeam"], table["AwayTeam"])):
            if pair in week:
                table.loc[idx, ["result", "predicted_score", "based_on"]] = week[pair]
    table["result"] = table["result"].astype(int)
    table["predicted_result"] = table["result"].map(RESULT_LABELS)
    table["predicted_winner"] = np.where(table["result"] == 2, table["HomeTeam"],
                                         np.where(table["result"] == 0, table["AwayTeam"], "Draw"))
    table = table.sort_values(["sort_date", "HomeTeam"], kind="mergesort").reset_index(drop=True)[SEASON_PRED_COLS]
    counts = table["predicted_result"].value_counts()
    print(f"Fixtures source: {source} | {len(table)} remaining fixtures in {int(X['round'].max()) + 1} rounds "
          f"| features built in {time.time() - _t:.0f}s")
    print(f"Models without odds: 1X2 {BEST_NO_ODDS_KEY} | goals {goals_key}")
    print("Predicted results: " + ", ".join(f"{counts.get(label, 0)} {label.lower()}s" for label in RESULT_LABELS.values()))
    return table


season_predictions = run_section("remaining-season predictions", predict_remaining_season)
if not isinstance(season_predictions, pd.DataFrame):
    season_predictions = pd.DataFrame(columns=SEASON_PRED_COLS)
season_predictions.to_csv(Path(CONFIG["out_dir"]) / "predictions_remaining_season.csv", index=False)
show(season_predictions.head(30), f"Predicted results for the rest of the season (first 30 of {len(season_predictions)})")

# %% [markdown]
# ### Saved artefacts

# %%
def save_artifacts() -> None:
    """Write the feature table and the final model bundle to the output directory."""
    out_dir = Path(CONFIG["out_dir"])
    try:
        features.to_parquet(out_dir / "features.parquet", index=False)
    except Exception as exc:
        print(f"Parquet write failed ({type(exc).__name__}); writing features.csv instead")
        features.to_csv(out_dir / "features.csv", index=False)
    joblib.dump({
        "classifier": FINAL_CLASSIFIER, "goals": FINAL_GOALS, "class_names": CLASS_NAMES,
        "feature_sets": FEATURE_SETS, "trained_through": str(DATA["Date"].max().date()),
        "selected_by": "walk-forward CV log loss on the development seasons",
        "holdout_metrics_dev_trained": RESULTS[BEST_KEY], "team_name_map": TEAM_NAME_MAP,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in CONFIG.items()},
        "usage": "probabilities = bundle_predict_proba(bundle['classifier'], feature_frame); columns are [Away, Draw, Home]",
    }, out_dir / "best_model.joblib")
    for name in ("features.parquet", "model_comparison.csv", "goals_regression_comparison.csv", "xg_experiment.csv",
                 "best_model.joblib", "predictions_next_gameweek.csv",
                 "predictions_remaining_season.csv"):
        path = out_dir / name
        print(f"  {'saved ' if path.exists() else 'absent'} {path}" + (f" ({path.stat().st_size / 1024:.0f} KB)" if path.exists() else ""))


save_artifacts()

# %% [markdown]
# ## 9. Results and limitations
#
# The cell below prints the headline numbers from this run. Read them against these expectations and caveats.
#
# **What to expect.** A good 1X2 model typically lands at 52 to 56 percent accuracy and only roughly matches
# the bookmakers. Closing odds aggregate team news, lineups and the money of well-informed bettors, none of
# which this pipeline sees. A model that appears to beat the closing line by a wide margin on the holdout
# should be treated as a bug or as luck until proven otherwise. The paired bootstrap interval in section 7.5
# is the honest summary: if it includes zero, the model and the market cannot be separated on this sample.
#
# **Limitations**
#
# - **Draws are hard.** About a quarter of matches are drawn, yet a draw is almost never the single most likely
#   outcome, so draw recall is close to zero for every model, including the bookmaker. Judge draw handling by
#   log loss, RPS and the reliability diagram rather than by recall.
# - **No injury, suspension or lineup data.** Squad news, managerial changes, transfers and European or cup
#   fixtures are invisible here. Rest days and congestion are computed from league matches only.
# - **Promoted teams are uncertain.** A promoted side has no recent top-flight history, so its early-season
#   features are borrowed from the previous season's bottom three and its Elo starts at their average. Expect
#   larger errors for these clubs until about ten matches have been played.
# - **Odds features dominate.** In feature set C the market probabilities carry most of the signal and the
#   model largely learns to copy the market, which is also why the value-bet backtest finds few bets there.
#   Sets A and B show what form and Elo can do alone.
# - **Risk of overfitting to a handful of seasons.** Model selection rests on six scored walk-forward seasons
#   and the holdout is two seasons (760 matches). Differences of a few thousandths in log loss between models
#   are well inside the noise, and the XGBoost search was tuned on some of the scored folds. The live block is
#   far too small to rank models.
# - **Structural change.** Home advantage, scoring rates and refereeing change over time (the 2020/21 season
#   without crowds is the clearest example). Old seasons help estimate stable effects but can also mislead.
# - **Data caveats.** The bookmaker benchmark uses closing odds only where they exist. Understat xG and ClubElo
#   are optional and may be missing on a given run (see the Data availability log). Optuna is time-boxed, so
#   tuned parameters can differ slightly between machines.
# - **Rest-of-season results are a rough guide.** `predictions_remaining_season.csv` freezes every club's form
#   at the latest result and uses no odds, so it cannot react to anything that happens in between. Re-run the
#   notebook each week. Only the next gameweek uses fresh form and market prices.
# - **Betting.** The backtest is illustrative. It ignores stake limits, price availability and odds movement.

# %%
def final_summary() -> None:
    """Headline numbers for this run."""
    best, bookmaker, no_odds = RESULTS[BEST_KEY], RESULTS[BOOKMAKER_KEY], RESULTS[BEST_NO_ODDS_KEY]
    print("FINAL SUMMARY")
    print(f"Holdout: {', '.join(season_label(s) for s in TEST_SEASONS)} ({int(MASK_TEST.sum())} matches). "
          f"Live block: {int(MASK_LIVE.sum())} matches.")
    for label, row in (("Best model", best), ("Best without odds", no_odds), ("Bookmaker", bookmaker), ("Class prior", RESULTS[PRIOR_KEY])):
        name = row["model"] + (f" [{row['feature_set']}]" if row["feature_set"] != "-" else "")
        print(f"  {label:<18} {name:<58} holdout log loss {row['test_logloss']:.4f} | RPS {row['test_rps']:.4f} "
              f"| accuracy {row['test_acc']:.1%} | draw recall {row['test_rec_draw']:.1%}")
    gap = best["test_logloss"] - bookmaker["test_logloss"]
    print(f"  Best model minus bookmaker (holdout log loss): {gap:+.4f} ({'model ahead' if gap < 0 else 'bookmaker ahead'}; "
          "see the bootstrap interval in section 7.5)")
    if isinstance(TOP_FEATURES, pd.DataFrame) and len(TOP_FEATURES):
        print(f"\nTop 10 features (XGBoost on {BEST_XGB_FSET}):")
        print(TOP_FEATURES.to_string(index=False))
    print(f"\nTotal runtime: {(time.time() - NOTEBOOK_T0) / 60:.1f} minutes")


final_summary()
show_comparison("Final metrics table")
show(pd.DataFrame(AVAILABILITY_LOG), "Data availability log")
