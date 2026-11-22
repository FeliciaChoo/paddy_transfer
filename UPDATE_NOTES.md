# Paddy mask update (PLANMalaysia polygons)

Unzip this over your project folder (same folder layout). 

## New files
- pipeline/paddy_mask.py            tags every sampled pixel against the paddy polygons (removes nothing)
- pipeline/paddy_robustness.py      repeats the yield tests on four pixel sets and writes a gated verdict
- tests/test_paddy_mask.py          9 tests (full suite: 58 pass)
- geo/paddy_planmalaysia.geojson    the polygons you supplied (10,022 polygons, Peninsular Malaysia)
- geo/granary_scheme_map.csv        granary -> scheme names (EDIT after the data owner confirms them)
- data/paddy_pixel_tags.csv, paddy_mask_summary.csv, paddy_scheme_area.csv, paddy_cleaning_log.csv,
  paddy_mask_correlation.csv, paddy_mask_growth_test.csv, paddy_mask_cold_start.csv, paddy_mask_verdict.md   (results already computed)
- docs/paddy_mask_assessment.md     method, results, limits

## Existing files that were CHANGED (merge by hand if you edited them)
- pipeline/eda_clean.py   additions only: new optional argument --paddy-tags, new function add_paddy_flags(), 3-line call. Without the option the output is byte-for-byte
                          the same columns as before (checked on your data). CRLF line endings kept.
- README.md               a 2b and a 6b step in the run section, and a new "Paddy mask" section before "Publishing"
- geo/README.txt          two entries appended
- requirements-pipeline.txt   one line: shapely

## Run (after pip install shapely)
python pipeline/paddy_mask.py --paddy geo/paddy_planmalaysia.geojson --pixels DailyGranary_by_granary.csv --out data
python pipeline/eda_clean.py --data DailyGranary_by_granary.csv --out eda --paddy-tags data/paddy_pixel_tags.csv     # optional flags
python pipeline/paddy_robustness.py --clean eda/DailyGranary_clean.csv --tags data/paddy_pixel_tags.csv --out data
python -m pytest tests -q

## Result: QUALITY_CONTROL_ONLY (see docs/paddy_mask_assessment.md)
Not changed: the dashboard (app.py, dashboard/), acts_transfer/, the models and every existing result file.

# PACE methodology update (paddy_transfer_methodology / pace_complete_clean_to_dashboard)

## New files
- pipeline/build_stage_features.py   A2 + C1: one row per granary-year, Savitzky-Golay green-up, C1_*/C2_* stage features   -> data/transfer_stage_dataset.csv, data/phenology_greenup.csv
- acts_transfer/_boost2.py           Two-Stage TrAdaBoost.R2 and S-TrAdaBoost.R2
- acts_transfer/screening.py         A3 domain-divergence screen of every candidate source granary, compatible pool
- tests/test_methodology_v2.py       21 tests (full suite: 83 pass)

## Changed (merge by hand if you edited them)
- acts_transfer/: config, methods (5 new methods, in-fold RobustScaler + VIF selection, aliases), evaluation (look-ahead guard, persistence, rRMSE/MAPE/win rate/bootstrap CI), modelchoice (Delta_NT safeguard),
  harmonise (pooled source, fold settings, shift_preview), selection (scenario G), predict, workflow, assessment (pooled units)
- dashboard/: tab_transfer (defaults MADA -> IADA BLS, compatible pool, screen, stage features, four-method panel, safeguard badge, LOYO/expanding-window, kg/ha metrics), tab_overview, about_config
- pipeline/build_transfer_dataset.py  ten-day table now also carries LSWI, MNDWI, Tmax
- tests/test_pipeline_order.py, tests/test_ui_gating.py  adapted (strict gate mode; default target is now IADA Barat Laut Selangor)

# PACE V4 update (pace_methodology_config_v4): evidence-first transfer qualification
- acts_transfer/sweep.py + pipeline/run_benchmark_sweep.py   10-granary benchmark sweep (MADA -> each other granary, LOYO + forward), pooled paired Delta_NT / Delta_DS, bootstrap CI (pairs and granaries),
  Wilcoxon, suitability mapping, qualification(). Results shipped in data/benchmark_sweep_{pairs,pooled,granary,suitability}.csv
- modelchoice.choose(..., sweep=): a transfer method is deployable only if the sweep passed for it AND Delta_NT < 0; table gains Delta_NT vs target-only, Delta_DS vs source-only, sweep qualified
- dashboard/tab_transfer.py: tab 5 now has Verify (sub-tab 1) and Discover (sub-tab 2); qualification panel; baselines labelled B1-B4; Optical_Valid_Fraction as a static covariate
- dashboard/about_config.py: Overview states the Discover -> Verify -> Deploy paradigm
- tests/test_methodology_v2.py: 25 tests (full suite 87)

# PACE V8 update (pace_methodology_config_v8): governance roles, Detrended Residual Transfer, sign-off
- acts_transfer/methods.py   DetrendedResidualTransfer + ridge_trend; acts_transfer/_boost2.py: initial row weights (optical quality)
- acts_transfer/modelchoice.py  dual-criteria gate: Delta_NT vs min(season trend, target-only, persistence) on COMMON seasons; domain relevance; min common seasons; deployable set (hybrid only)
- acts_transfer/evaluation.py   strict past-only persistence in LOYO; train RMSE/MAE and test-train gap in metrics; optical weights passed to the methods
- acts_transfer/governance.py   registry (results/governance_registry.json), states, sign-off; acts_transfer/publish.py: what a sign-off publishes for the KPKM views
- acts_transfer/scenarios.py    Normal / El Nino drought / Wet monsoon feature scenarios; predict.py: bands widen with the horizon
- dashboard/tab_governance.py   overview, replay, nowcast + outlook, sign-off; tab_advisory.py: two-tab brief (Who/What/When/Why, field checklist, CSV); tab_map.py: Optical_Valid_Fraction layer
- app.py                        Workspace Role + Selected Granary sidebar, five tabs per role
- results/ (empty registry: every granary starts PENDING_REVIEW; nothing is pre-approved)
- data/benchmark_sweep_*.csv re-run with the hybrid and heading-stage features

# PACE V9 update (the three Gemini files: map, overview hub, app.py)
- app.py                       replaced by the V9 sidebar navigation (Overview / KPKM workspace 4 pages / Validator workflow 4 steps with Next-Back) and the botanical theme
- dashboard/tab_map.py         + render_leaflet_map(D, role): Leaflet map, GeoJSON polygons, click-to-select, optical-coverage colouring for the validator, pydeck fallback
- dashboard/tab_overview.py    + render_overview_hub(D)
- dashboard/tab_yield.py, tab_advisory.py, tab_transfer.py   + the entry points the new app.py calls: render_yield_view, render_decision_brief, render_domain_audit, render_benchmark_audit, render_validator_signoff
- pipeline/build_granary_geojson.py + data/malaysia_granaries.geojson   (the file the map needs; it did not exist)
- requirements.txt             + folium, streamlit-folium, branca
- tests/test_ui_gating.py rewritten for the sidebar navigation; tests/test_map_v9.py new
Changes I made to the supplied files: the overview and map read the growth-stage table (the yield table has no Optical_Valid_Fraction), the governance registry is read through acts_transfer.governance
(its file is {"granaries": {...}}, not a flat dict), the granary list is the data's (KETARA, not "IADA KETARA"), the yield legend classes come from the data range (2.4 to 5.7 t/ha, not the fixed 1.87 to 4.81),
a map click is applied once so it cannot override a later sidebar choice, buttons use width="stretch", and the sidebar CSS keeps select-box text readable.

# PACE V10 update (gemini-code V10 .py + pace V10 json)
- app.py                 PaddyTwin sidebar: tagline, workspace cards, page / step cards with a lime active outline, Selected Granary, role card; ivory + olive-lime theme; shorter sidebar step labels,
                         full step titles as page headings (the V10 file's sidebar dropped the granary selector; I kept it, every page needs it)
- dashboard/tab_map.py   the map is now centroid CircleMarkers with a sticky HTML tooltip (district, state, yield, optical coverage, season windows); click -> nearest marker -> Selected Granary.
                         The V9 polygon map, data/malaysia_granaries.geojson and pipeline/build_granary_geojson.py are removed. Yield classes: from the data range, or the fixed V10 legend (radio).
- data/granary_meta.csv  State, Season_1_Window, Season_2_Window per granary (from the V10 file; unverified). Marker positions: data/districts.csv, not the coordinates typed in the V10 file (within 0.1 degree of them)
- data/advisory_reference_practice.csv + about_config.ADVISORY_KNOWLEDGE_SOURCES   the cited reference practice cards in the Decision Brief (from the V10 file; unverified)
- dashboard/tab_overview.py   + Mean Optical Coverage KPI, 'Enter ...' button labels. Kept the label 'Mean Granary Yield': the V10 file calls the simple mean 'Area-Weighted', but no planted area is available.
- tests: 114 pass (UI tests follow the new sidebar labels; map tests cover markers, click matching, tooltip, legend, metadata, reference practice)

# Map spec update (interactive_granary_leaflet_map)
- tab_map.py: only the default base layer is shown (folium showed all three base layers at once, so the satellite layer covered the default map: fixed); fixed widget key 'leaflet_granary_map';
  click read from last_object_clicked_tooltip (nearest-marker position as a fallback); field summary card under the map (yield, change on previous year, rank, optical coverage, state, windows, governance state);
  no pydeck fallback any more (a table is shown if folium is missing); the fixed 1.87-4.81 t/ha legend is now the DEFAULT (data-range classes remain a radio option); out-of-range granaries are listed under the map
- data/granary_meta.csv: the season text and marker positions of this spec (Marker_Lat, Marker_Lon). This reverses my V10 choice of pixel-centre positions: the specified positions are used, pixel centres are the fallback
- width: st_folium's width is an integer, so use_container_width=True (full width) is used instead of width='100%'
