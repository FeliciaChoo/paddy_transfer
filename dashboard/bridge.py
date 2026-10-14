"""The bridge between the model results and the people who act on them (KPKM / granary administrators).

    model results (forecast, range, status, yield outlook, data quality)
        -> case file for one granary                         build_case()
        -> a confidence level, with the reasons               confidence()
        -> administrative interventions (who / when / why)    interventions()   <- editable rules, data/intervention_rules.csv
        -> numbered evidence list for the AI memo             evidence()        <- the AI may use ONLY this
        -> the human's decision (accept / modify / reject)    log_decision()    <- data/intervention_log.csv

Nothing here calls a model. It only translates results the pipeline already produced, so every line an administrator
sees can be traced back to a number, a rule id, or a recorded decision.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from . import adminmap
from .ui import pct

LEVELS = ["low", "medium", "high"]
RULE_COLS = ["rule_id", "priority", "status", "trend", "stage_contains", "flag", "needs_confidence",
             "action", "owner", "timing", "why", "review_status"]
LOG_COLS = ["time", "granary", "rule_id", "action", "decision", "note", "decided_by", "data_as_of", "status", "confidence"]

# Draft rules for administrators. They only say what to CHECK, ASK or PREPARE. They never name a cause, dose or product,
# because satellite greenness cannot diagnose one. Edit data/intervention_rules.csv to change them; an agronomist should review.
DEFAULT_RULES = [
    ["R01", 1, "behind", "any", "any", "any", "any",
     "Send a field verification team to {granary} and record planting dates, water supply, flooding, pests and disease for the affected blocks.",
     "Granary authority with the state Department of Agriculture", "Within 7 days",
     "Recent satellite greenness has been below the range expected for this stage; the satellite cannot say why.", "DRAFT"],
    ["R02", 2, "behind", "any", "tillering|stem|panicle|booting|germination", "any", "any",
     "Ask the irrigation and drainage unit to confirm water supply and scheduling for {granary} during this growth stage.",
     "Granary irrigation / drainage unit", "Within 7 days",
     "Water supply is one possible reason for weak growth at this stage; confirm it before looking at other causes.", "DRAFT"],
    ["R03", 2, "ahead", "any", "any", "any", "any",
     "Confirm planting dates and varieties in {granary} with the granary office before reading early greenness as good news.",
     "Granary authority", "Within 2 weeks",
     "Greener than expected can come from earlier planting or a different crop stage, not only from a better crop.", "DRAFT"],
    ["R04", 2, "any", "any", "maturity|grain|heading|flowering", "any", "any",
     "Prepare harvest logistics for {granary}: harvester availability, drying and storage capacity, and transport to buyers.",
     "Granary authority with paddy buyers and mills", "Start now; harvest is expected within about 30 to 60 days",
     "The crop is estimated to be at {stage} in 30 days (from the planting calendar, not observed).", "DRAFT"],
    ["R05", 3, "any", "any", "fallow|turnaround|germination|establishment", "any", "any",
     "Check readiness for the next planting cycle in {granary}: seed supply, land-preparation machinery and the irrigation start date.",
     "Granary authority", "Before planting begins",
     "The crop is estimated to be at {stage} in 30 days (from the planting calendar, not observed).", "DRAFT"],
    ["R06", 3, "any", "falling", "tillering|stem|panicle|booting", "any", "medium",
     "Ask the granary office whether any block in {granary} was hit by flood, drought or pests, because greenness is falling before the crop should peak.",
     "Granary authority", "Within 2 weeks",
     "A falling NDVI trend during the growth stages is not the normal pattern; the cause is unknown.", "DRAFT"],
    ["R07", 1, "any", "any", "any", "stale", "any",
     "Refresh the satellite extraction (use a later --end-year) before acting on this granary; the data describes the past, not today.",
     "KPKM data team", "Before any resource decision",
     "The newest satellite observation is {stale_days} days old ({as_of}).", "DRAFT"],
    ["R08", 2, "any", "any", "any", "yield_jump", "any",
     "Ask the granary authority how yield was counted in {years} and confirm it before using the yield outlook for planning.",
     "KPKM statistics unit with the granary authority", "Before the next planning cycle",
     "Year-to-year jumps of 20% or more often reflect a change in counting, not the crop.", "DRAFT"],
    ["R09", 3, "any", "any", "any", "shared_series", "any",
     "This granary reports the same yield figures as {partners}; request separate yield reporting if you need separate decisions.",
     "KPKM statistics unit", "Next reporting round",
     "Two granaries with an identical yield series are not independent evidence.", "DRAFT"],
    ["R10", 2, "any", "any", "any", "no_sat", "any",
     "Extend satellite coverage to {granary}; only yield history exists for it.",
     "KPKM data team", "Next data refresh", "There is no satellite forecast for this granary.", "DRAFT"],
    ["R11", 3, "any", "any", "any", "weak_signal", "any",
     "Obtain the official planting calendar for {granary}; the satellite shows no clear planting cycle, so stage estimates are unreliable.",
     "Granary authority", "Next planning cycle",
     "Planting dates look mixed within this granary, so the estimated stage may be wrong.", "DRAFT"],
    ["R12", 1, "any", "any", "any", "low_confidence", "any",
     "Do not allocate extra resources on this alert alone; combine it with field reports first.",
     "KPKM programme manager", "Before approving any spending",
     "Confidence is low: {conf_reasons}.", "DRAFT"],
    ["R13", 4, "on track", "any", "any", "any", "any",
     "Routine monitoring only for {granary}; no extra resources are indicated by the current evidence.",
     "Granary authority", "Next monthly review",
     "Recent greenness is inside the range expected for this stage.", "DRAFT"],
]


def default_rules():
    return pd.DataFrame(DEFAULT_RULES, columns=RULE_COLS)


def ensure_rules_file(data_dir):
    """write the default rules once; after that the CSV is the source of truth (edit it)"""
    p = data_dir / "intervention_rules.csv"
    if not p.exists():
        default_rules().to_csv(p, index=False)
    return p


def matches(cell, value):
    c = str(cell).strip().lower()
    return c == "any" or str(value).lower() in [x.strip().lower() for x in c.split("|")]


def stage_matches(cell, stage):
    c = str(cell).strip().lower()
    return c == "any" or any(k.strip() in str(stage).lower() for k in c.split("|"))


# ----------------------------------------------------------------------------------------------------------------------
# 1. results -> one case file per granary
# ----------------------------------------------------------------------------------------------------------------------
def granary_skill(D, g):
    """this granary's own NDVI forecast error against 'no change', in the last held-out year (None if too few points)"""
    p = D.preds[(D.preds.District == g) & (D.preds.test_year == D.FINAL_TEST_YEAR)].dropna(subset=["observed", "forecast"])
    out = {}
    for h, q in p.groupby("horizon"):
        if len(q) >= 10:
            m, b = (q.forecast - q.observed).abs().mean(), (q.ndvi_last - q.observed).abs().mean()
            out[int(h)] = None if b == 0 else float(1 - m / b)
    return out


def transfer_note(D):
    """one honest sentence about the transfer-learning benchmark (the benchmark must have been run: pipeline/transfer_benchmark.py)"""
    t = getattr(D, "tl_summary", None)
    if t is None or t.empty:
        return None, "Transfer learning (TrAdaBoost.R2) has not been benchmarked in this run, so the forecasts use normal training."
    r = t[(t.horizon == 1) & (t.k_years == t.k_years.min())]
    if r.empty:
        return None, "Transfer learning was benchmarked but the summary has no row to compare."
    r = r.iloc[0]
    rel = (r.tradaboost - r.pooled_ridge) / r.pooled_ridge * 100
    if rel > 5:
        return False, (f"Transfer learning (TrAdaBoost.R2) was tested against normal training on every granary and was NOT better "
                       f"(its forecast error was {rel:.0f}% higher with {int(r.k_years)} year of own data), so these forecasts use "
                       "normal pooled training.")
    if rel < -5:
        return True, (f"Transfer learning (TrAdaBoost.R2) reduced forecast error by {abs(rel):.0f}% against normal training with "
                      f"{int(r.k_years)} year of own data.")
    return None, "Transfer learning (TrAdaBoost.R2) made no clear difference against normal training, so normal training is used."


def build_case(D, g, today=None):
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    c = D.cond[D.cond.District == g]
    s = c.iloc[0] if len(c) else None
    gy = D.yh[D.yh.District == g].sort_values("Year")
    o = D.outlook[D.outlook.District == g].sort_values("Horizon_years")
    pts = adminmap.load()
    admin = ""
    if pts is not None and g in set(pts[1].District):
        admin = pts[1].set_index("District").Admin[g]
    case = dict(granary=g, admin=admin, has_sat=s is not None, today=today)
    if s is not None:
        age = int((today - pd.Timestamp(s.As_of)).days)
        case.update(status=s.Recent_status, trend=s.Trend, stage=s.Stage_30d, signal=s.Signal, as_of=str(s.As_of),
                    stale_days=age, last_ndvi=float(s.Last_NDVI), f30=float(s.Forecast_30d), lo30=float(s.Lower_30d),
                    hi30=float(s.Upper_30d), below=int(s.Recent_below), above=int(s.Recent_above),
                    checked=int(s.Recent_checked), below_share=float(s.Below_share_10d), amplitude=float(s.NDVI_amplitude))
    if len(gy):
        last = gy.iloc[-1]
        case.update(yield_year=int(last.Year), yield_last=float(last.Yield), yield_change=None if pd.isna(last.Change_pct)
                    else float(last.Change_pct), yield_z=float(last.Yield_z), yield_mean=float(D.MEAN_YIELD[g]),
                    jump_years=[int(y) for y in gy[gy.Flag_large_jump].Year],
                    partners=str(gy.Shares_series_with.iloc[0]) if isinstance(gy.Shares_series_with.iloc[0], str) else "")
    if len(o):
        n = o.iloc[0]
        case.update(outlook_year=int(n.Forecast_year), outlook=float(n.Central), outlook_low=float(n.Low),
                    outlook_high=float(n.High), outlook_band=float(n.Band_pct), outlook_method=str(n.Method))
    case["skill"] = granary_skill(D, g)
    case["tl_adopted"], case["tl_text"] = transfer_note(D)
    case["confidence"], case["conf_reasons"] = confidence(case)
    case["flags"] = flags(case)
    return case


def confidence(case):
    """high / medium / low for the SATELLITE alert, with the reasons. Transparent thresholds, no hidden score."""
    if not case["has_sat"]:
        return "low", ["no satellite forecast exists"]
    lvl, why = 2, []

    def cap(n, text):
        nonlocal lvl
        if n < lvl:
            lvl = n
        why.append(text)

    if case["stale_days"] > 90:
        cap(0, f"the newest satellite data is {case['stale_days']} days old")
    elif case["stale_days"] > 45:
        cap(1, f"the newest satellite data is {case['stale_days']} days old")
    sk = case["skill"].get(3)
    if sk is None:
        cap(1, "too few held-out decades to judge this granary's own forecast skill")
    elif sk <= 0:
        cap(0, f"this granary's 30-day forecast was no better than 'no change' last year ({sk:+.0%})")
    elif sk < 0.15:
        cap(1, f"this granary's 30-day forecast beat 'no change' by only {sk:.0%} last year")
    if case["signal"] == "weak":
        lvl = max(0, lvl - 1)
        why.append("planting dates look mixed, so the stage estimate is uncertain")
    return LEVELS[lvl], why or ["recent data and a forecast that beat 'no change' for this granary"]


def flags(case):
    f = []
    if not case["has_sat"]:
        f.append("no_sat")
    else:
        if case["stale_days"] > 45:
            f.append("stale")
        if case["signal"] == "weak":
            f.append("weak_signal")
    if case.get("jump_years"):
        f.append("yield_jump")
    if case.get("partners"):
        f.append("shared_series")
    if case["confidence"] == "low":
        f.append("low_confidence")
    return f


# ----------------------------------------------------------------------------------------------------------------------
# 2. case -> interventions (rules) -> evidence for the AI
# ----------------------------------------------------------------------------------------------------------------------
class _Safe(dict):
    def __missing__(self, k):
        return "n/a"


def interventions(case, rules):
    """matching rules, most urgent first: [(action, owner, timing, why, rule_id, review_status)]"""
    fill = _Safe(granary=case["granary"], stage=case.get("stage", "n/a"), as_of=case.get("as_of", "n/a"),
                 stale_days=case.get("stale_days", "n/a"), years=", ".join(map(str, case.get("jump_years", []))),
                 partners=case.get("partners", ""), conf_reasons="; ".join(case["conf_reasons"]))
    status, trend, stage = case.get("status", "n/a"), case.get("trend", "n/a"), case.get("stage", "")
    hits = []
    for _, r in rules.iterrows():
        flag_ok = str(r.flag).strip().lower() == "any" or str(r.flag).strip().lower() in case["flags"]
        conf_ok = str(r.needs_confidence).strip().lower() in ("any", "") or \
            LEVELS.index(case["confidence"]) >= LEVELS.index(str(r.needs_confidence).strip().lower())
        sat_rule = str(r.status).lower() != "any" or str(r.trend).lower() != "any" or str(r.stage_contains).lower() != "any"
        if sat_rule and not case["has_sat"]:
            continue
        if flag_ok and conf_ok and matches(r.status, status) and matches(r.trend, trend) and stage_matches(r.stage_contains, stage):
            hits.append(r)
    hits = sorted(hits, key=lambda r: (int(r.priority), r.rule_id))
    # routine monitoring is only useful when nothing more urgent applies
    if any(int(r.priority) <= 3 for r in hits):
        hits = [r for r in hits if r.rule_id != "R13"]
    return [(str(r.action).format_map(fill), r.owner, r.timing, str(r.why).format_map(fill), r.rule_id, r.review_status)
            for r in hits]


def evidence(D, case):
    """numbered evidence items. The AI memo may use ONLY these; the memo's numbers are checked against them."""
    ev = []
    if case["has_sat"]:
        ev += [("E1", f"Crop status of {case['granary']} on {case['as_of']}: {case['status']}; "
                      f"{case['below']} of the last {case['checked']} decades were below the forecast range and {case['above']} above."),
               ("E2", f"NDVI was {case['last_ndvi']:.2f}; the 30-day forecast is {case['f30']:.2f} (80% range {case['lo30']:.2f} to "
                      f"{case['hi30']:.2f}); trend: {case['trend']}. Estimated stage in 30 days: {case['stage']} (from the planting "
                      "calendar, not observed)."),
               ("E3", f"In {D.FINAL_TEST_YEAR}, {pct(case['below_share'])} of decades were below the 10-day forecast range "
                      "(about 10% is expected by chance).")]
    else:
        ev.append(("E1", f"No satellite forecast exists for {case['granary']}; only yield history is available."))
    if "yield_last" in case:
        t = (f"Reported yield {case['yield_year']}: {case['yield_last']:,.0f} kg/ha; the granary's average is "
             f"{case['yield_mean']:,.0f} kg/ha.")
        if "outlook" in case:
            t += (f" Baseline outlook {case['outlook_year']}: {case['outlook']:,.0f} kg/ha (+/-{case['outlook_band']:.0f}%, "
                  f"method: {case['outlook_method']}).")
        ev.append(("E4", t))
    dq = []
    if case.get("jump_years"):
        dq.append("year-to-year yield jumps of 20% or more in " + ", ".join(map(str, case["jump_years"])))
    if case.get("partners"):
        dq.append(f"the yield series is identical to {case['partners']}")
    if case["has_sat"] and case["signal"] == "weak":
        dq.append("planting dates look mixed")
    ev.append(("E5", "Data quality: " + ("; ".join(dq) if dq else "no yield jump or shared series flagged") + "."))
    sk = case["skill"]
    if sk:
        ev.append(("E6", f"In {D.FINAL_TEST_YEAR} this granary's NDVI forecast beat 'no change' by "
                         + " and ".join(f"{pct(v)} at {10 * h} days" for h, v in sorted(sk.items()) if v is not None) + "."))
    ev.append(("E7", f"Confidence in the satellite alert: {case['confidence']}. Reasons: {'; '.join(case['conf_reasons'])}."))
    ev.append(("E8", case["tl_text"]))
    return ev


def memo_task(case, actions):
    guide = "\n".join(f"- {a[0]} (owner: {a[1]}; timing: {a[2]}; why: {a[3]})" for a in actions) or "- No rule matched."
    task = (f"Write a one-page decision memo for KPKM programme administrators about the granary '{case['granary']}'"
            + (f" (administrative area: {case['admin']})" if case["admin"] else "") + ". Sections: What is happening to the paddy; "
            "How confident we are; Recommended interventions (keep each owner and timing, add a 'Why:' line); What to verify "
            "before acting. Base the interventions ONLY on this approved guidance and the evidence:\n" + guide)
    return task


# ----------------------------------------------------------------------------------------------------------------------
# 3. the human's decision goes back into a log (the start of a validation record for the alerts)
# ----------------------------------------------------------------------------------------------------------------------
def log_path(data_dir):
    return data_dir / "intervention_log.csv"


def read_log(data_dir):
    p = log_path(data_dir)
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLS)
    return pd.read_csv(p)


def log_decision(data_dir, case, action, rule_id, decision, note, by):
    row = dict(time=datetime.now().strftime("%Y-%m-%d %H:%M"), granary=case["granary"], rule_id=rule_id, action=action,
               decision=decision, note=note.strip(), decided_by=by.strip(), data_as_of=case.get("as_of", ""),
               status=case.get("status", ""), confidence=case["confidence"])
    p = log_path(data_dir)
    pd.concat([read_log(data_dir), pd.DataFrame([row])], ignore_index=True)[LOG_COLS].to_csv(p, index=False)
    return row
