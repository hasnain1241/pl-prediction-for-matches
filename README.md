# Premier League match prediction

A machine learning pipeline that predicts English Premier League results from historical data, plus a
small static website that shows its predictions and how each model scored.

This was built purely for practice, mainly to see how XGBoost works on a real problem. It is a
statistics project, not betting advice.

## What it does

The notebook downloads results from football-data.co.uk (2000/01 onward), builds leakage-safe features
(recent form, Elo ratings and, in one feature set, bookmaker odds), and trains several models:

- classifiers for home win, draw and away win: logistic regression, random forest, XGBoost, LightGBM,
  with calibrated and ensemble versions
- goals models (Poisson GLM and XGBoost) turned into result probabilities with a Dixon-Coles scoreline grid
- baselines, including the bookmaker's closing odds

Validation is time-ordered and nothing is shuffled. Models are chosen by walk-forward validation on the
development seasons. The 2024/25 and 2025/26 seasons are a holdout used once for the final numbers.

## Headline results

- **No model beat the bookmaker.** The best model's holdout log loss is 0.9959. Closing odds score
  0.9893 (lower is better).
- Holdout accuracy is about 50 to 52 percent.
- Draws are almost never predicted. The latest run predicts none of the remaining 330 matches as a draw.
- No injury or lineup data is used.

The full table is in `model_comparison.csv` and on the website's Model performance page.

## Repository layout

| Path | Contents |
| --- | --- |
| `code/epl_match_prediction.ipynb` | The pipeline as a notebook |
| `code/epl_match_prediction.py` | The same pipeline as a script |
| `*.csv` in the root | Outputs of the latest run |
| `docs/` | The static website served by GitHub Pages |
| `docs/data/` | Copies of the CSVs that the website loads, plus `meta.json` |

## Run the notebook on Kaggle

1. Create a new Kaggle notebook and upload `code/epl_match_prediction.ipynb`.
2. In the sidebar, open **Settings** and switch **Internet** on so the notebook can download the season files.
3. Choose **Run All**.
4. When it finishes, download the output files from `/kaggle/working`: `predictions_next_gameweek.csv`,
   `predictions_remaining_season.csv`, `model_comparison.csv`, `goals_regression_comparison.csv` and `xg_experiment.csv`.

## Refresh the website

The site is plain HTML, CSS and JavaScript with no build step. GitHub Pages cannot run Python, so the
site only displays the CSV files.

1. Re-run the notebook.
2. Copy the four CSVs the site reads into `docs/data/`: `predictions_remaining_season.csv`,
   `model_comparison.csv`, `goals_regression_comparison.csv` and `xg_experiment.csv`.
3. Set `trained_through` in `docs/data/meta.json` to the date of the latest result in the data. The
   notebook prints it as "Latest result in the data".
4. Commit and push. GitHub Pages redeploys within a minute or two.

```bash
cp predictions_remaining_season.csv model_comparison.csv goals_regression_comparison.csv xg_experiment.csv docs/data/
git add docs/data
git commit -m "Refresh predictions"
git push
```

To preview the site locally:

```bash
python3 -m http.server 8000 -d docs
```

Then open http://localhost:8000. Opening the HTML files directly does not work because the browser
blocks `fetch` on `file://` pages.

## Data sources

- [football-data.co.uk](https://www.football-data.co.uk/) for results, match statistics and bookmaker odds
- [Understat](https://understat.com/) for expected goals
- The [vaastav Fantasy-Premier-League archive](https://github.com/vaastav/Fantasy-Premier-League)

## Licence

MIT. See [LICENSE](LICENSE).
