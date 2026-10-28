# ACTS granary dashboard (for KPKM / granary administrators)

Streamlit dashboard for paddy granaries. The unit everywhere is the **granary**, drawn on the maps as the administrative
district(s) it covers. Eight tabs:

| # | Component | File |
|---|-----------|------|
| 1 | Overview (the About page; names and credits are edited in `dashboard/about_config.py`) | `dashboard/tab_overview.py` |
| 2 | Dynamic maps | `dashboard/tab_map.py` |
| 3 | Yield and outlook: two separate sections (Yield: what was reported; Outlook: what to expect), each with its own map and ranking and a button that shows the last records and trend | `dashboard/tab_yield.py` |
| 4 | Advisory and intervention: press **Get advisory**, record KPKM's decisions, download the Word report | `dashboard/tab_advisory.py`, `dashboard/report.py`, built on `dashboard/bridge.py` |
| 5 | Decision tracker: every decision, follow-up status, and how KPKM's choices differ from the advice | `dashboard/tab_tracker.py`, `dashboard/tracker.py` |
| 6 | Model check: results without transfer learning, a button that applies it, and the improvement | `dashboard/tab_modelcheck.py` |

## Run it (PowerShell, one command per line, in this order)

```powershell
pip install -r requirements-pipeline.txt      # the dashboard alone needs only requirements.txt

# 1. satellite extraction (Earth Engine) -> DailyGranary.csv  (District + Granary columns). Slow, resumable, cached in gee_cache/
python pipeline/granary_direct.py --yield-csv malaysia_granary_annual_ml_ready.csv --out DailyGranary.csv --end-year 2026

# 2. granary-level table (the pipeline and the app treat the 'District' column as the unit)
python pipeline/make_granary_level.py --daily DailyGranary.csv

# 3. EDA and cleaning, 4. NDVI forecast model, 5. app data
python pipeline/eda_clean.py --data DailyGranary_by_granary.csv --out eda
python pipeline/final_model.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out final_model
python pipeline/prepare_app_data.py --yield-csv malaysia_granary_annual_by_granary.csv --pheno eda/eda_phenology.csv --final-dir final_model --clean eda/DailyGranary_clean.csv --out data

# 6. evidence + outlook + resilience (all write into data/, which the dashboard reads)
python pipeline/transfer_benchmark.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out data   # NDVI: transfer vs normal training
python pipeline/yield_dynamics_transfer.py --out data        # yield: transfer of yield dynamics, back-test (also gives the bands)
python pipeline/yield_outlook_transfer.py --out data         # yield outlook for the dashboard (re-run it whenever prepare_app_data.py is re-run)
python pipeline/resilience_from_forecast.py --clean eda/DailyGranary_clean.csv --out data   # resilience through the NDVI forecast + weather/soil context

# 7. dashboard
streamlit run app.py
```

## AI memo (Gemini): the key lives on the server, never in the app
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put the key in it (or set the `GEMINI_API_KEY`
environment variable; on Streamlit Cloud use App settings -> Secrets). Nothing is typed into the app. Without a key the
whole app works; only the AI memo is skipped. `secrets.toml` is in `.gitignore`: never commit a key.

## Map boundaries
`geo/` holds the district and state boundaries and `granary_districts.csv`, the editable list of which districts belong
to which granary. Boundary source and a licence reminder are in `geo/README.txt`.

## How a model result reaches a decision (`dashboard/bridge.py`)
model result -> evidence E1..E8 -> confidence (high / medium / low, with reasons) -> intervention rules
(`data/intervention_rules.csv`: who, when, why) -> AI memo written only from that evidence -> the administrator's
decision (accept / modify / reject / defer) saved to `data/decision_tracker.csv` (see below).

## Transfer learning: the result
`pipeline/transfer_benchmark.py` tests every granary as a "new" granary with 1 to 3 years of its own satellite history,
against normal training (pooled Ridge, pooled gradient boosting, and a plain AdaBoost control with the same weak learner).
On this data TrAdaBoost.R2 was worse than normal training in every case, so the forecasts use normal pooled training.
Re-run it after new data arrives; the Advisory tab reports whatever it finds.

## Folder layout
```
app.py                  entry point (header, sidebar, 4 tabs)
dashboard/              data.py (loads data/), mapkit.py + adminmap.py (maps), bridge.py (results -> interventions),
                        llm.py (server-side Gemini + number check), ui.py, charts.py, one tab_*.py per component
pipeline/               extraction, granary table, EDA, forecast model, app data, transfer benchmark
geo/                    administrative boundaries and the granary-district table
.streamlit/             secrets.toml.example
data/                   generated; intervention_rules.csv is written once and is then yours to edit
```

## What the numbers say (so nobody is misled)
- **NDVI forecast** (10 and 30 days): trained model, beats "no change" in most granaries. Used for crop status and **resilience**.
- **Resilience**: measured through the forecast (how often/deeply NDVI falls below the expected range, how fast it recovers, how steady the
  peak is). Relative to these granaries; it showed **no relationship with yield stability** (n = 10).
- **Yield outlook**: transfer learning of yield dynamics (carry-over rates learned on the other granaries). It beats training on one granary
  alone (about 1.8 points of error at 1 year) but only **matches** "same as last year" (about 8% error at 1 year, 11% at 2, 14% at 3).
  Satellite, weather and resilience inputs did not improve it (`pipeline/yield_transfer_benchmark.py`, `yield_dynamics_transfer.py`).
- **TrAdaBoost** on NDVI and on yield inputs did not beat normal training, so it is not used.

## Sources for the advisory
`dashboard/knowledge.py` holds the reference library (ids M1, W1, T1, ...) and a guide to every collected feature. Each rule in
`data/intervention_rules.csv` has a `sources` column. The same tables are shown in the Advisory tab and exported to `evidence/`.
Rules stay DRAFT until an agronomist reviews them.

## The Advisory tab and the report
The page shows nothing until KPKM presses **Get advisory for <granary>**. It then shows that granary's summary and the recommended
interventions (priority, who, by when, why). Two buttons download an editable Word report: one granary, or all granaries ranked for
attention. The report has the summary, the interventions, a decision and planning table for KPKM to fill in, a box and sign-off lines
for the final conclusion, and the basis, confidence, evidence and sources (`dashboard/report.py`). The workings (rules, evidence,
references) are not shown on the page; they are in the report appendix.

## Recording and tracking decisions
After **Get advisory**, KPKM records a decision for each proposed intervention: **Accept** (as proposed), **Modify** (say what will be done
instead; required), **Reject** (give the reason; required) or **Defer**, plus the responsible officer, target date and status, and an
overall next step for the granary. Each save is appended to the database, so the history is kept; an item's current state is its latest
row. A decision belongs to the advice for one data date: when the data is refreshed the new advice starts as Pending. The **Decision
tracker** tab shows everything recorded, lets status, officer, date and remarks be updated as work progresses, flags overdue items, and
compares accepted, modified and rejected counts per intervention type so weak rules can be reviewed. The Word report is prefilled with the
recorded decisions and the overall conclusion.

## Database
Decisions are stored with SQLAlchemy (`dashboard/store.py`). Nothing to set up: by default it creates the SQLite file `data/acts.db` on
first use, and imports an old `decision_tracker.csv` once if there is one. To use another database, set `DATABASE_URL` in
`.streamlit/secrets.toml` (or as an environment variable); the table `decision_tracker` is created for you:

```toml
DATABASE_URL = "postgresql+psycopg2://user:password@host:5432/dbname"   # pip install psycopg2-binary
# DATABASE_URL = "mysql+pymysql://user:password@host:3306/dbname"        # pip install pymysql
```
Use an external database (PostgreSQL, MySQL) for several users or for Streamlit Community Cloud; a local SQLite file is erased there when
the app restarts. The tracker tab shows where the data is stored (the password is hidden).

**Looking at the data in PyCharm.** PyCharm Professional: View > Tool Windows > Database, then + > Data Source > SQLite, choose
`data/acts.db`, download the driver if asked, Test Connection, and open the table `decision_tracker`. PyCharm Community has no Database tool:
open the same file with the free "DB Browser for SQLite", or query it from Python with `sqlite3`.
**Running the app from PyCharm.** Run > Edit Configurations > + > Python; Module name `streamlit`; Parameters `run app.py`; Working
directory: the project folder; Python interpreter: the project `.venv`.

## Maps: the panel beside the map
On every map (Overview, Dynamic maps, Yield) the space beside the map holds the legend and the information for the granary you click: status,
stage, NDVI trend, confidence, resilience, yield and outlook, a small chart and the most urgent intervention. It scrolls inside its own box,
so the page does not need scrolling (`dashboard/sidepanel.py`).

## Model check (tab 6)
Step 1 shows the yield outlook without transfer learning (each granary's own history). The button **Apply transfer learning** repeats the
back-test with the carry-over rate learned on the other granaries and shows the change, with a 95% range and the share of cases improved.
It runs live (about half a second). TrAdaBoost is too slow to repeat there, so its stored results (`transfer_benchmark.py`,
`yield_transfer_benchmark.py`) are shown next to it. `pipeline/yield_dynamics_transfer.py` holds the back-test (`run_backtest`).

## Why 10 and 30 days, and what was tested
The satellite series is built in 10-day blocks (36 a year), so 10 days is one step ahead and 30 days is three. The dashboard shows these two
(`--horizons 1 3` in `pipeline/final_model.py`); other horizons run with the same flag. Tested out to 90 days, the model's average error stays
low (0.060 NDVI at 10 days, 0.072 at 30, 0.079 at 90) because it knows the planting cycle, so 30 days is a design choice, not a limit found in
the data. Beyond 30 days the forecast is mostly the typical curve for the planting date. `pipeline/yield_instance_weighting.py` tests
instance-weighting transfer learning (TrAdaBoost, similarity weighting, up-weighting the granary's own years) on the yield dynamics: none beat
the current method (results in the Model check tab).

## Yield and outlook tab (tab 3)
The tab has two sections. **Yield** shows reported yield for a chosen past year (map, ranking, the selected granary's figures beside the map).
**Outlook** shows the projection for the next 1 to 4 years with its range and typical error. Neither section shows crop condition, advice or
other details. In each, the button **Show the last records and trend** opens the selected granary's records as a chart and a table (the
Outlook version adds the outlook to the chart); it stays closed until pressed.

## Overview (About) page
`dashboard/tab_overview.py`: what the dashboard is, four questions it answers, what you can do, how it works, who it is for, data and methods,
important notes (the satellite end date is computed from the data), research credits, and the reliability notes. Change the app name,
tagline and the Research and development lines in `dashboard/about_config.py` (the credit lines are placeholders to fill in).

## Transfer-learning assessment (scripts and results)
`pipeline/pre_transfer_assessment.py` (is it safe to transfer, and from which granaries: covariates only), `pipeline/transfer_methods.py` (feature-based,
parameter-based and source-selection methods on the NDVI task), `pipeline/yield_dynamics_methods.py` (nearest-source and shrinkage for the yield
outlook), `pipeline/transfer_error_diagnosis.py` (why some methods did worse). Run them in that order after step 6 with `--out data`. The written
assessment is `docs/transfer_learning_assessment.md`; the expectations written BEFORE the methods were run are `data/pre_transfer_expectations.md`.

## Publishing
See `DEPLOY.md` (Streamlit Community Cloud from a private GitHub repository, shared with your supervisor by email).

## Transfer-learning workflow (tabs 6 and 7)
Tab **6 · Pre-Transfer Assessment** (select domains, assess data, harmonise, evaluate suitability) must be completed before tab **7 · Transfer Learning**
(select strategy, run, compare). The order is enforced in the backend (`acts_transfer/workflow.py`), not only on the page. The method, formulas, scenarios,
validation and results are in `docs/transfer_workflow_guide.md`; two complete example runs are in `docs/example_run_BLS_to_MADA.md` and
`docs/example_run_BLS_to_KETARA.md`. Tests: `python -m pytest tests -q`.
