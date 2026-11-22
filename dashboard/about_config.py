"""Text you may want to change on the Overview (About) page. Edit this file only; nothing else needs to change."""

APP_NAME = "Granary Monitor"
TAGLINE = "Turning satellite, weather and yield data into decisions for the paddy granaries"
CLOSING = "Granary Monitor: from agricultural data to informed decisions."

# Shown under "Research and development". Replace the bracketed parts with your own details; delete a line to remove it.
RESEARCH = [
    "Project: [project title]",
    "Developer: [your name], [programme], [faculty]",
    "[University]",
    "Supervisor: [supervisor name and title]",
    "Contact: [email address]",
]

# Project scope shown on the Overview (PLAN stage of PACE: Plan, Analyze, Construct, Execute)
PROJECT_TITLE = "Cross-Granary Paddy Yield Prediction and Intervention Decision Support System"
RESEARCH_PROBLEM = ("IADA Barat Laut Selangor (Sekinchan) has only 7 yearly yield records (2018 to 2024), too few to train a reliable local model. "
                    "The project asks whether yield-response knowledge from data-abundant MADA (Kedah), and the climate-compatible Malaysian granaries, can be transferred to it. "
                    "MADA is not assumed to be a good source: it is qualified empirically (Discover, then Verify, then Deploy), and transfer is refused wherever it would make the forecast worse (negative transfer).")
PARADIGM = "Evidence-first transfer qualification: Discover (10-granary benchmark sweep) → Verify (this target, safeguard gate) → Deploy (forecast, intervals, advisory)."
SCOPE_ROWS = [
    ("Paradigm", PARADIGM),
    ("Candidate pair", "MADA, Kedah  →  IADA Barat Laut Selangor (Sekinchan), nt = 7. Qualified only after the benchmark sweep and the Negative-Transfer Safeguard Gate (Delta_NT < 0)."),
    ("Source pool", "MADA alone, or the compatible pool of 8 granaries. KETARA is screened out (its climate is almost perfectly separable from the target: domain AUC above 0.9); the 8-source and 9-source pools can be compared."),
    ("Granaries mapped", "All 10 Malaysian granaries; selecting one on the map carries it to the later tabs."),
    ("Unit of analysis", "One row per granary and year: 10 granaries × 7 years = 70 rows (60 for 2019 to 2024, when optical coverage is complete). Pixels are never rows."),
    ("Target variable", "Yield_kg_ha"),
    ("Reference baselines", "B1 Target-Only (normal training) · B2 Source-Only (pretrain and test) · B3 Naive pooled training · B4 Persistence and target-mean floors"),
    ("Validation", "Leave-one-year-out and forward expanding-window; no random or spatial splits."),
    ("Safety rule", "Delta_NT = RMSE(transfer) − min(RMSE target-only Ridge, RMSE persistence). Below zero (and the sweep passed) the transfer model is deployed; otherwise the system falls back to a local baseline."),
]
PACE_ROWS = [
    ("PLAN", "1 Overview", "Scope, stakeholders, unit of analysis"),
    ("ANALYZE", "2 Dynamic maps · 3 Data Assessment", "Cloud audit, one row per granary-year, domain-divergence screen"),
    ("CONSTRUCT", "3 Data Assessment · 4 Transfer Strategy", "Phenology stage features, fold-safe harmonisation, transfer models and baselines"),
    ("EXECUTE: Discover", "5 Train and Evaluate, sub-tab 2", "10-granary benchmark sweep: pooled paired Delta_NT and Delta_DS, bootstrap interval, Wilcoxon, suitability mapping"),
    ("EXECUTE: Verify", "3 and 5 Train and Evaluate, sub-tab 1", "Select the target, run LOYO and forward validation, apply the safeguard gate"),
    ("EXECUTE: Deploy", "6 Yield Prediction · 7 Future Yield Outlook · 8 Intervention Advisory", "Forecast (kg/ha) with ranges, growth-stage drivers, advisory, report, decision log"),
]

# Knowledge sources behind the reference practice shown in the Decision Brief (PACE V10). Supplied with the V10 specification; not verified by this project.
ADVISORY_KNOWLEDGE_SOURCES = [
    "KPKM Staggered Paddy Planting Schedule (Jadual Penanaman Padi) & MYSA MakGeoPadi",
    "DOA Manual Tanaman Padi & Kaedah Pengurusan Tanaman Padi (Kejuruteraan Pertanian)",
    "IADA Pulau Pinang Paddy Management Manual & SBPKP/SIPP Subsidy Guidelines",
    "FAO Irrigation and Drainage Paper (Chapter 4: Paddy Rice Irrigation Scheduling)",
    "PACE V10 Detrended Residual Transfer Model (Hybrid Season Trend + S-TrAdaBoost.R2)",
]
