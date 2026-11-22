# ACTS granary dashboard (for KPKM / granary administrators)

Streamlit dashboard for paddy granaries. The unit everywhere is the **granary**, drawn on the maps as the administrative
district(s) it covers. Eight tabs:

| # | Tab | What it does | Code |
|---|-----|--------------|------|
| 1 | Overview | what the dashboard is, notes, reliability | `dashboard/tab_overview.py` |
| 2 | Dynamic maps | interactive maps of yield and crop condition | `dashboard/tab_map.py` |
| 3 | Data Assessment | select source and target, assess, harmonise, say whether transfer is suitable | `dashboard/tab_transfer.py` (`render_assessment`) |
| 4 | Transfer Strategy | turn the assessment's evidence into a recommended method; review and confirm | `render_strategy` |
| 5 | Train and Evaluate Model | train the strategy and the baselines, compare, say whether transfer helps, pick the model of record | `render_training` |
| 6 | Yield Prediction | predictions from the selected model, each labelled with the model that made it (+ reported-yield view) | `render_prediction` |
| 7 | Future Yield Outlook | the outlook from the selected model and stated input assumptions, against the baseline outlook | `render_outlook` |
| 8 | Intervention Advisory | advice that uses the validated prediction when there is one (+ decision tracker) | `dashboard/tab_advisory.py` |

Order: **Overview → Dynamic maps → Data Assessment → Transfer Strategy → Train and Evaluate Model → Yield Prediction → Future Yield Outlook → Intervention Advisory.**
It is enforced in the backend (`acts_transfer/workflow.py`). A change upstream clears everything that depends on it and the page says what was cleared; tabs 6 and 7
recalculate from the new model.

## Run it (PowerShell, one command per line, in this order)

```powershell
pip install -r requirements-pipeline.txt      # the dashboard alone needs only requirements.txt

# 1. satellite extraction (Earth Engine) -> DailyGranary.csv  (District + Granary columns). Slow, resumable, cached in gee_cache/
python pipeline/granary_direct.py --yield-csv malaysia_granary_annual_ml_ready.csv --out DailyGranary.csv --end-year 2026

# 2. granary-level table (the pipeline and the app treat the 'District' column as the unit)
python pipeline/make_granary_level.py --daily DailyGranary.csv

# 2b. OPTIONAL paddy mask: tags every sampled pixel against the PLANMalaysia paddy polygons (removes nothing; see "Paddy mask" below)
python pipeline/paddy_mask.py --paddy geo/paddy_planmalaysia.geojson --pixels DailyGranary_by_granary.csv --out data

# 3. EDA and cleaning, 4. NDVI forecast model, 5. app data
python pipeline/eda_clean.py --data DailyGranary_by_granary.csv --out eda
# (to add the paddy flags to the cleaned file: append  --paddy-tags data/paddy_pixel_tags.csv  to the line above)
python pipeline/final_model.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out final_model
python pipeline/prepare_app_data.py --yield-csv malaysia_granary_annual_by_granary.csv --pheno eda/eda_phenology.csv --final-dir final_model --clean eda/DailyGranary_clean.csv --out data

# 5b. PACE methodology: granary-year table with phenology-aligned growth-stage features (C1_*/C2_*), then the ten-day driver table (adds LSWI, MNDWI, Tmax)
python pipeline/build_stage_features.py --clean eda/DailyGranary_clean.csv --out data      # -> data/transfer_stage_dataset.csv (N = 70), data/phenology_greenup.csv
python pipeline/build_transfer_dataset.py --clean eda/DailyGranary_clean.csv --out data    # -> data/transfer_decade_features.csv (for the domain-divergence screen)
python pipeline/run_benchmark_sweep.py --data data --out data                              # -> data/benchmark_sweep_*.csv (PACE V4 Discover; about a minute)

# 6. evidence + outlook + resilience (all write into data/, which the dashboard reads)
python pipeline/transfer_benchmark.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out data   # NDVI: transfer vs normal training
python pipeline/yield_dynamics_transfer.py --out data        # yield: transfer of yield dynamics, back-test (also gives the bands)
python pipeline/yield_outlook_transfer.py --out data         # yield outlook for the dashboard (re-run it whenever prepare_app_data.py is re-run)
python pipeline/resilience_from_forecast.py --clean eda/DailyGranary_clean.csv --out data   # resilience through the NDVI forecast + weather/soil context

# 6b. OPTIONAL: does restricting the satellite features to real paddy change the yield result? (writes data/paddy_mask_*.csv and a verdict)
python pipeline/paddy_robustness.py --clean eda/DailyGranary_clean.csv --tags data/paddy_pixel_tags.csv --out data

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

## Paddy mask (PLANMalaysia polygons)
`pipeline/paddy_mask.py` cleans the PLANMalaysia paddy layer (`geo/paddy_planmalaysia.geojson`: 10,022 polygons, 366,191 ha, Peninsular Malaysia only) and tags
each of the 751 sampled pixels as on paddy or not, inside a granary scheme or not, inside its own granary's scheme or not. Nothing is deleted: the tags go to
`data/paddy_pixel_tags.csv`, and `eda_clean.py --paddy-tags` can add them to the cleaned file as `flag_not_paddy` and `flag_outside_own_scheme`.
`pipeline/paddy_robustness.py` then repeats three tests on four pixel sets and writes a verdict (`data/paddy_mask_verdict.md`).

**Result on this data: QUALITY_CONTROL_ONLY.** 86% of the sampled pixels are on mapped paddy, but restricting the satellite features to paddy does not make
them predict yield: the year-to-year (within-granary) correlation stays near zero under every mask and no masked set beats "same as last year". Use the mask
to document and audit the pixel sample, not as an accuracy improvement. Details, tables and limits: `docs/paddy_mask_assessment.md`.
The scheme labels in the layer look incomplete (for example only ~9,400 ha is labelled KADA): `geo/granary_scheme_map.csv` is yours to edit once the data owner confirms them.

## Publishing
See `DEPLOY.md` (Streamlit Community Cloud from a private GitHub repository, shared with your supervisor by email).


## PACE V10 navigation, and V8 governance

The sidebar is the PaddyTwin layout: a tagline, **YOUR WORKSPACE** cards (Overview, KPKM Programme Manager, Model Validator & Data Steward), the pages or audit steps of the chosen workspace as bordered cards
(lime outline on the active one), the shared **Selected Granary** (sidebar, or click a marker on a map), and a role card at the bottom. The palette is warm ivory with botanical olive-lime (defined in `app.py`).

| Workspace | Pages |
|---|---|
| Overview | Executive Overview & Governance Status: KPIs (including Mean Optical Coverage and Verified Transfer Models n / 10), a summary card per workspace with an *Enter* button, and the national snapshot with each granary's registry state |
| KPKM Programme Manager | Granary Field Condition Map · Historical Replay & Yield Fit · In-Season Nowcast & 2025–2028 Outlook · Programme Manager Decision Brief (*Understand the results* / *Plan next steps*, with a cited reference-practice section) |
| Model Validator & Data Steward | Step 1 Spatial & Optical Quality Map · Step 2 Data & Domain Divergence Audit · Step 3 Transfer Testing & Train/Test Audit · Step 4 Validator Sign-Off & Assurance Gate, with Next / Back buttons that scroll to the top |

The Leaflet map (`dashboard/tab_map.render_leaflet_map`; needs folium and streamlit-folium, the pydeck map is the fallback) draws one **circle marker per granary at the centre of its sampled pixels**, so it needs no
shape file. Hover shows district, state, yield (t/ha and kg/ha), optical coverage and the two KPKM planting windows; a click selects the granary. Markers are coloured by yield with the fixed 1.87 to 4.81 t/ha legend of the specification by default (a radio switches to five classes over the data range, because the data reach 5.7 t/ha; granaries outside the
fixed range are listed under the map); the validator can also colour by optical valid coverage. A click (read from the marker's tooltip text) selects the granary and a field summary card appears under the map.
State, planting windows and marker positions are in `data/granary_meta.csv`; a granary with no position there is placed at the centre of its sampled pixels (`data/districts.csv`).
The reference practice in the Decision Brief is `data/advisory_reference_practice.csv` (who, topic, window, action, source). **Both the planting windows and the reference practice were supplied with the V10 specification and have not been checked by this project: verify them against the KPKM schedule and the DOA / IADA manuals before relying on them.**

The validator writes `results/governance_registry.json`; the KPKM role only reads it and never runs or configures a model. A granary with no entry is **PENDING_REVIEW** and is served by the local baseline.
States: **APPROVED** (green "Verified Transfer Model"), **REJECTED_LOCAL_FALLBACK** and **PENDING_REVIEW** (amber "Local Baseline Active"). Approval is refused unless the stored gate evidence shows both criteria passed.

- **Detrended Residual Transfer** (the only method that can be approved): a regularised linear trend per granary (fitted in the training fold only) carries the yield drift; S-TrAdaBoost.R2, weighted by
  w = max(0.2, Optical_Valid_Fraction), transfers only the residual anomaly from MADA; prediction = trend + residual. The other transfer methods are benchmarks.
- **Dual-criteria gate:** (1) domain relevance, no flagged granary (KETARA) in the source pool; (2) Delta_NT = RMSE_transfer − min(RMSE_season_trend, RMSE_target_only, RMSE_persistence) < 0, with every RMSE taken
  over the **seasons common to the transfer model and all three baselines** (the trend needs 3 training years, so it is scored on fewer seasons; comparing different seasons is not fair), and at least 3 common seasons.
- **Leak-free splits:** forward expanding-window is primary; leave-one-year-out is diagnostic and its persistence is strictly past-only (the first year has no persistence forecast).
- **Heading-stage features only** by default (vegetative and reproductive windows): ripening starts after the 70-day nowcast. Cycle 2 is opt-in.
- **Nowcast and outlook:** the nowcast uses the heading-stage features plus the trend (a back-test of the latest season, since satellite data ends 2024-12); the 2025–2028 outlook uses the trend plus a scenario
  (Normal, El Niño drought, Wet monsoon = the dry/hot or wet tail of the year-to-year anomalies seen across the granaries), with a band that widens with the horizon. Flat historical-average inputs are never used alone.
- **Overfitting audit:** train RMSE next to test RMSE for every method (tab 4), and a sensitivity check that drops the weakest optical year and the weights.

## Transfer-learning workflow (tabs 3 to 7), PACE methodology

Paradigm (PACE V4): **evidence-first transfer qualification, Discover → Verify → Deploy.** MADA is not assumed to be a good source for IADA Barat Laut Selangor; it is qualified empirically.

- **Discover** (tab 5, sub-tab 2): the benchmark sweep. MADA is transferred to each of the other 9 granaries under identical out-of-fold splits (about 63 granary-years under LOYO). Paired error differences are pooled:
  Delta_NT = error(transfer) − error(Target-Only), Delta_DS = error(transfer) − error(Source-Only), in kg/ha (yield levels are never averaged across granaries). Tests: paired bootstrap 95% interval
  (over pairs, and over granaries as a cautious check) and the Wilcoxon signed-rank test. A suitability table relates each granary's Delta_NT to domain AUC, MMD², CORAL, calendar offset, yield CV and nt
  (Spearman; 9 granaries, so descriptive). MADA cannot be its own target, so the sweep has 9 targets, not 10. A method **passes the sweep** when the pooled Delta_NT interval is below 0 and the one-sided Wilcoxon p < 0.05.
- **Verify** (tab 5, sub-tab 1, with tab 3): the target's own out-of-fold Delta_NT and its rank among the sweep granaries. A transfer model is **qualified** only if the sweep passed for that method AND the safeguard gate (Delta_NT < 0) passes.
- **Deploy** (tabs 6 to 8): only a qualified transfer model drives the forecast; otherwise the best local baseline does, and the tabs say so.
- Baselines are labelled B1 Target-Only (normal training), B2 Source-Only (pretrain and test), B3 Naive pooled, B4 Persistence and target-mean floors; Standard TrAdaBoost.R2 stays as a diagnostic reference.
- Soil_pH is constant within a granary, so the sweep drops it when it does not vary in the target's own years; Optical_Valid_Fraction is kept as a static covariate.

Primary benchmark: **MADA (Kedah) → IADA Barat Laut Selangor (Sekinchan)**, target with 7 yearly records. The tabs open on these defaults. The unit is one row per granary-year
(10 granaries × 7 years = 70 rows; 60 for 2019–2024). Pixels are never rows.

| PACE | Step | Where |
|---|---|---|
| Analyze A2 | pixel-decades → one row per (District, Year); pixel mean/std/p10/p90 of EVI kept as spatial descriptors | `pipeline/build_stage_features.py` |
| Analyze A3 | every candidate granary screened against the target: domain-classifier AUC, MMD², CORAL, KS, calendar offset; a granary with climate AUC ≥ 0.90 (KETARA) is flagged and left out of the **compatible pool** (8 sources); the pool can be run with it to compare 8 and 9 sources | `acts_transfer/screening.py`, tab 3 Step 2 |
| Construct C1 | Savitzky–Golay (window 5, order 2) on the 36 ten-day EVI values → green-up of Cycle 1 (main) and Cycle 2 (off-season) → Vegetative 0–40 d (EVI, MNDWI, SAR VV), Reproductive 40–70 d (EVI, LSWI, SAR VV, Rain, Tmax), Ripening 70–100 d (EVI, Solar) + Soil_pH | `pipeline/build_stage_features.py` |
| Construct C2 | **inside each training fold only:** median imputation, RobustScaler, within-granary Z-score (anomalies), CORAL alignment, VIF/correlation selection (p ≤ 6) | `acts_transfer/methods.py` (`Prep`), `harmonise.py` |
| Construct C3/C4 | Granary-Intercept Ridge, KMM + Weighted Ridge, Two-Stage TrAdaBoost.R2, S-TrAdaBoost.R2, Z-Score Anomaly Transfer against persistence, target-only Ridge, source-only, pooled; **LOYO** and **forward expanding-window** validation | `methods.py`, `_boost2.py`, `evaluation.py` |
| Execute E1 | RMSE, MAE (kg/ha), rRMSE, MAPE, R², win rate vs persistence, bootstrap 95% intervals | `evaluation.py` |
| Execute E2 | **Negative-transfer safeguard:** Delta_NT = RMSE(transfer) − min(RMSE target-only Ridge, RMSE persistence). Below 0 → deploy; otherwise automatic fallback to the best **local** baseline (never a source-trained model) | `modelchoice.py`, badge in tabs 5–7 |

Decisions that are easy to miss:

- **Look-ahead guard.** A pooled source holds other granaries in the same years as the held-out target year. Source rows of the held-out year (and, in forward validation, any later year)
  are not used for training (`Config.source_time_safe`, on by default).
- **Safeguard strictness.** The default follows the methodology: Delta_NT < 0 deploys. With 4 to 5 held-out years that can be a tiny, noisy margin, so the page labels such a result
  "statistically not supported". `Config.nt_gate = "delta_nt+stats"` (or the radio button in tab 3 → Thresholds) additionally requires the bootstrap interval to exclude zero and the result to hold under LOYO.
  The older five-criterion gate is that stricter mode, and its tests still run.
- **Cycle 2 (off-season) features are opt-in.** The off-season cycle is detected in only 32 of 70 granary-years (it is missing in 86% of MADA rows), so the readiness check blocks it; the main-season C1_* features are the default.
- S-TrAdaBoost.R2 is implemented from the published idea (importance sampling, balanced weights), not as a line-by-line port of the authors' code.

Tab 4 recommends methods from the assessment's evidence (scenarios A to G; G = yield-level difference between granaries) and has a button that adds the four methodology methods.
Tab 5 trains them with the baselines and shows the safeguard. **Transfer learning becomes the model used for predictions only if the safeguard passes**; otherwise a local baseline stays the
model of record and tabs 6 to 8 say so. Details, formulas and results: `docs/transfer_workflow_guide.md`. Tests: `python -m pytest tests -q` (see tests/).
