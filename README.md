# ACTS granary dashboard (for KPKM / granary administrators)

Streamlit dashboard for paddy granaries. The unit everywhere is the **granary**, drawn on the maps as the administrative
district(s) it covers. Four components:

| # | Component | File |
|---|-----------|------|
| 1 | Overview | `dashboard/tab_overview.py` |
| 2 | Dynamic maps | `dashboard/tab_map.py` |
| 3 | Yield: past values (map + graph) and the outlook | `dashboard/tab_yield.py` |
| 4 | Advisory and intervention | `dashboard/tab_advisory.py`, built on `dashboard/bridge.py` |

## Run it (PowerShell, one command per line, in this order)

```powershell
pip install -r requirements.txt

# 1. satellite extraction (Earth Engine) -> DailyGranary.csv  (District + Granary columns). Slow, resumable, cached in gee_cache/
python pipeline/granary_direct.py --yield-csv malaysia_granary_annual_ml_ready.csv --out DailyGranary.csv --end-year 2026

# 2. granary-level table (the pipeline and the app treat the 'District' column as the unit)
python pipeline/make_granary_level.py --daily DailyGranary.csv

# 3. EDA and cleaning, 4. NDVI forecast model, 5. app data
python pipeline/eda_clean.py --data DailyGranary_by_granary.csv --out eda
python pipeline/final_model.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out final_model
python pipeline/prepare_app_data.py --yield-csv malaysia_granary_annual_by_granary.csv --pheno eda/eda_phenology.csv --final-dir final_model --clean eda/DailyGranary_clean.csv --out data

# 6. is transfer learning (TrAdaBoost.R2) worth it? Writes data/transfer_*.csv, which the Advisory tab reports
python pipeline/transfer_benchmark.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out data

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
decision (accept / modify / reject / defer) saved to `data/intervention_log.csv`.

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
