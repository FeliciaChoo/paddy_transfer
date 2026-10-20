"""The bridge between the model results and the people who act on them (KPKM / granary administrators).

    model results (forecast, range, status, yield outlook, data quality)
        -> case file for one granary                         build_case()
        -> a confidence level, with the reasons               confidence()
        -> administrative interventions (who / when / why)    interventions()   <- editable rules, data/intervention_rules.csv
        -> numbered evidence list for the AI memo             evidence()        <- the AI may use ONLY this
        -> the human's decision (accept / modify / reject)    see dashboard/tracker.py  <- data/decision_tracker.csv

Nothing here calls a model. It only translates results the pipeline already produced, so every line an administrator
sees can be traced back to a number, a rule id, or a recorded decision.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from . import adminmap
from .ui import pct

LEVELS = ["low", "medium", "high"]
# Thresholds behind the weather / soil flags. 'z' = standard deviations from this granary's own normal for the same time of year.
HEAT_C = 33.0          # mean daily maximum over the last 3 decades; the literature puts 33 C as where production starts to fall (T1)
WARM_NIGHT_Z = 1.5     # unusually warm nights (T2, T3)
LOW_SOLAR_Z = -1.0     # unusually little sunshine (L1, L2): only matters in the reproductive / ripening stages (see the rule)
WET_Z, DRY_Z = 1.5, -1.5
ACID_PH = 5.8          # flooded-soil iron-toxicity watch level (P3); the map pH is only an estimate
RULE_COLS = ["rule_id", "priority", "status", "trend", "stage_contains", "flag", "needs_confidence",
             "action", "owner", "timing", "why", "review_status", "sources"]

# Draft rules for administrators. They only say what to CHECK, ASK or PREPARE. They never name a cause, dose or product,
# because satellite greenness cannot diagnose one. Edit data/intervention_rules.csv to change them; an agronomist should review.
DEFAULT_RULES = [
    ["R01", 1, "behind", "any", "any", "any", "any",
     "Send a field verification team to {granary} and record planting dates, water supply, flooding, pests and disease for the affected blocks.",
     "Granary authority with the state Department of Agriculture", "Within 7 days",
     "Recent satellite greenness has been below the range expected for this stage; the satellite cannot say why.", "DRAFT", "M1,M3,M4,W1"],
    ["R02", 2, "behind", "any", "tillering|stem|panicle|booting|germination", "any", "any",
     "Ask the irrigation and drainage unit to confirm water supply and scheduling for {granary} during this growth stage.",
     "Granary irrigation / drainage unit", "Within 7 days",
     "Water supply is one possible reason for weak growth at this stage; confirm it before looking at other causes.", "DRAFT", "W1,W3,W2"],
    ["R03", 2, "ahead", "any", "any", "any", "any",
     "Confirm planting dates and varieties in {granary} with the granary office before reading early greenness as good news.",
     "Granary authority", "Within 2 weeks",
     "Greener than expected can come from earlier planting or a different crop stage, not only from a better crop.", "DRAFT", "S1,G1"],
    ["R04", 2, "any", "any", "maturity|grain|heading|flowering", "any", "any",
     "Prepare harvest logistics for {granary}: harvester availability, drying and storage capacity, and transport to buyers.",
     "Granary authority with paddy buyers and mills", "Start now; harvest is expected within about 30 to 60 days",
     "The crop is estimated to be at {stage} in 30 days (from the planting calendar, not observed).", "DRAFT", "G1,G2"],
    ["R05", 3, "any", "any", "fallow|turnaround|germination|establishment", "any", "any",
     "Check readiness for the next planting cycle in {granary}: seed supply, land-preparation machinery and the irrigation start date.",
     "Granary authority", "Before planting begins",
     "The crop is estimated to be at {stage} in 30 days (from the planting calendar, not observed).", "DRAFT", "M1,L3,M2"],
    ["R06", 3, "any", "falling", "tillering|stem|panicle|booting", "any", "medium",
     "Ask the granary office whether any block in {granary} was hit by flood, drought or pests, because greenness is falling before the crop should peak.",
     "Granary authority", "Within 2 weeks",
     "A falling NDVI trend during the growth stages is not the normal pattern; the cause is unknown.", "DRAFT", "W3,M3"],
    ["R07", 1, "any", "any", "any", "stale", "any",
     "Refresh the satellite extraction (use a later --end-year) before acting on this granary; the data describes the past, not today.",
     "KPKM data team", "Before any resource decision",
     "The newest satellite observation is {stale_days} days old ({as_of}).", "DRAFT", ""],
    ["R08", 2, "any", "any", "any", "yield_jump", "any",
     "Ask the granary authority how yield was counted in {years} and confirm it before using the yield outlook for planning.",
     "KPKM statistics unit with the granary authority", "Before the next planning cycle",
     "Year-to-year jumps of 20% or more often reflect a change in counting, not the crop.", "DRAFT", ""],
    ["R09", 3, "any", "any", "any", "shared_series", "any",
     "This granary reports the same yield figures as {partners}; request separate yield reporting if you need separate decisions.",
     "KPKM statistics unit", "Next reporting round",
     "Two granaries with an identical yield series are not independent evidence.", "DRAFT", ""],
    ["R10", 2, "any", "any", "any", "no_sat", "any",
     "Extend satellite coverage to {granary}; only yield history exists for it.",
     "KPKM data team", "Next data refresh", "There is no satellite forecast for this granary.", "DRAFT", "S2"],
    ["R11", 3, "any", "any", "any", "weak_signal", "any",
     "Obtain the official planting calendar for {granary}; the satellite shows no clear planting cycle, so stage estimates are unreliable.",
     "Granary authority", "Next planning cycle",
     "Planting dates look mixed within this granary, so the estimated stage may be wrong.", "DRAFT", "S1,L3"],
    ["R12", 1, "any", "any", "any", "low_confidence", "any",
     "Do not allocate extra resources on this alert alone; combine it with field reports first.",
     "KPKM programme manager", "Before approving any spending",
     "Confidence is low: {conf_reasons}.", "DRAFT", ""],
    ["R13", 4, "on track", "any", "any", "any", "any",
     "Routine monitoring only for {granary}; no extra resources are indicated by the current evidence.",
     "Granary authority", "Next monthly review",
     "Recent greenness is inside the range expected for this stage.", "DRAFT", ""],
    ["R14", 2, "any", "any", "heading|flowering|panicle|booting|grain", "heat", "any",
     "Warn the granary authority that daily maximum temperatures in {granary} recently averaged {max_temp} C while the crop is near flowering; check for poor grain set and, for the next cycle, review the planting schedule so flowering avoids the hottest weeks.",
     "Granary authority with the state Department of Agriculture", "Within 2 weeks",
     "Heat around flowering raises spikelet sterility; production starts to fall above about 33 C.", "DRAFT", "T1,T4,T2"],
    ["R15", 3, "any", "any", "any", "warm_nights", "any",
     "Record the run of unusually warm nights in {granary} and ask the research station to check grain set at the next sampling.",
     "Granary authority with the research station", "Next sampling round",
     "Warm nights have been linked to lower yield in historical data.", "DRAFT", "T2,T3"],
    ["R16", 2, "any", "any", "heading|flowering|panicle|booting|grain|maturity", "low_solar", "any",
     "Brief KPKM that sunlight in {granary} has been well below its normal while the crop is forming grain, so a lower yield is plausible; for the next cycle review the planting schedule so these stages avoid cloudy months.",
     "KPKM programme manager with the granary authority", "Next planning cycle",
     "Low sunlight in the reproductive and ripening stages lowers yield.", "DRAFT", "L1,L2,L3"],
    ["R17", 2, "any", "any", "any", "wet_excess", "any",
     "Ask the drainage unit to confirm that drains and pumps serving {granary} are ready for the rainfall seen, and to report any flooded blocks.",
     "Granary drainage unit", "Within 7 days",
     "Rainfall has been far above this granary's normal for the time of year.", "DRAFT", "W1,M2"],
    ["R18", 2, "any", "any", "any", "dry_deficit", "any",
     "Ask the irrigation unit to confirm water supply to {granary} and to protect water for fields around flowering.",
     "Granary irrigation unit", "Within 7 days",
     "Rainfall has been far below normal; a water deficit around flowering costs more yield than one early in the season.", "DRAFT", "W1,W3"],
    ["R19", 4, "any", "any", "any", "acid_soil", "any",
     "Plan a soil test for {granary} before the next land preparation (topsoil pH is about {ph}) and review liming and potassium with the result.",
     "Granary authority with the Department of Agriculture", "Before the next planting",
     "Low pH in flooded soils is linked to iron toxicity; the map pH is only an estimate.", "DRAFT", "P1,P2,P3,M1"],
    ["R20", 1, "behind", "any", "any", "low_resilience", "any",
     "Escalate: {granary} is in the lower resilience third (it dips below the expected range more often, or recovers more slowly, than the other granaries). Bring the verification visit forward and request a follow-up satellite review in 10 days.",
     "KPKM programme manager with the granary authority", "Within 3 days",
     "A granary with weaker resilience is less likely to come back on its own.", "DRAFT", "R1,R2"],
    ["R21", 1, "behind", "any", "any", "slow_recovery", "any",
     "Treat the current shortfall in {granary} as unusual: it is longer than any in this granary's record (longest past run {longest} decade(s)). Request a written ground report within 7 days.",
     "Granary authority", "Within 7 days",
     "{below} of the last 4 decades are below the expected range; past shortfalls cleared in about {typ_rec} days.", "DRAFT", "R1,R2"],
]


def default_rules():
    return pd.DataFrame(DEFAULT_RULES, columns=RULE_COLS)


def ensure_rules_file(data_dir):
    """write the default rules once; after that the CSV is the source of truth (edit it)"""
    p = data_dir / "intervention_rules.csv"
    if p.exists() and "sources" not in pd.read_csv(p, nrows=1).columns:        # an older rules file: keep it, then upgrade
        p.replace(data_dir / "intervention_rules_old.csv")
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
    rf, cx = getattr(D, "resil_fc", None), getattr(D, "ctx", None)
    if rf is not None and g in set(rf.District):
        r = rf.set_index("District").loc[g]
        case.update(res_class=str(r.Resilience_class), res_index=float(r.Resilience_index), shock_share=float(r.Shock_share),
                    shock_depth=float(r.Shock_depth), rec_decades=float(r.Recovery_decades), longest_run=int(r.Longest_run),
                    typ_rec=int(r.Typical_recovery_days))
    if cx is not None and g in set(cx.District):
        c_ = cx.set_index("District").loc[g]
        case["ctx"] = dict(max_temp=float(c_.Mean_Max_Temp_C), min_z=float(c_.Mean_Min_Temp_C_z), rain_z=float(c_.Rain_mm_z),
                           solar_z=float(c_.Solar_Radiation_MJ_z), ph=float(c_.Soil_pH), max_z=float(c_.Mean_Max_Temp_C_z))
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
    if case.get("res_class") == "lower":
        f.append("low_resilience")
    if case["has_sat"] and case.get("status") == "behind" and "longest_run" in case and case["below"] > case["longest_run"]:
        f.append("slow_recovery")
    x = case.get("ctx")
    if x:
        for name, hit in (("heat", x["max_temp"] >= HEAT_C), ("warm_nights", x["min_z"] >= WARM_NIGHT_Z),
                          ("low_solar", x["solar_z"] <= LOW_SOLAR_Z), ("wet_excess", x["rain_z"] >= WET_Z),
                          ("dry_deficit", x["rain_z"] <= DRY_Z), ("acid_soil", x["ph"] < ACID_PH)):
            if hit:
                f.append(name)
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
                 partners=case.get("partners", ""), conf_reasons="; ".join(case["conf_reasons"]),
                 max_temp=f"{case['ctx']['max_temp']:.1f}" if case.get("ctx") else "n/a",
                 ph=f"{case['ctx']['ph']:.1f}" if case.get("ctx") else "n/a", longest=case.get("longest_run", "n/a"),
                 below=case.get("below", "n/a"), typ_rec=case.get("typ_rec", "n/a"), res_class=case.get("res_class", "n/a"))
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
    out = []
    for r in hits:
        src = r.get("sources", "")
        out.append((str(r.action).format_map(fill), r.owner, r.timing, str(r.why).format_map(fill), r.rule_id, r.review_status,
                    "" if pd.isna(src) else str(src), int(r.priority)))
    return out


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
    if "res_class" in case:
        ev.append(("E9", f"Resilience of {case['granary']} measured through the NDVI forecast (relative to the other granaries, not an absolute "
                         f"standard): {case['res_class']} third, index {case['res_index']:.0f} of 100. In held-out forecasts "
                         f"{pct(case['shock_share'])} of decades fell below the expected range (average shortfall {case['shock_depth']:.2f} NDVI) "
                         f"and a shortfall usually lasted about {case['typ_rec']} days (longest run {case['longest_run']} decade(s)). "
                         "This index has not been shown to predict yield."))
    x = case.get("ctx")
    if x:
        ev.append(("E10", f"Weather and soil over the last 3 decades before {case.get('as_of', 'the data end')}: daily maximum temperature "
                          f"{x['max_temp']:.1f} C, rainfall {x['rain_z']:+.1f} SD, sunshine {x['solar_z']:+.1f} SD and night temperature "
                          f"{x['min_z']:+.1f} SD from this granary's normal for the time of year; topsoil pH about {x['ph']:.1f} (map estimate)."))
    return ev


PRIORITY_LABEL = {1: "Urgent", 2: "Soon", 3: "Plan ahead", 4: "Routine"}


def priority_label(p):
    return PRIORITY_LABEL.get(int(p), "Routine")


def summary_text(D, case):
    """the granary summary in plain sentences (no model, no AI): what is happening, how sure, what the yield is doing"""
    g, out = case["granary"], []
    if case["has_sat"]:
        s = case["status"]
        if s == "behind":
            out.append(f"As of {case['as_of']}, {case['below']} of the last {case['checked']} ten-day periods in {g} were below the "
                       "greenness range expected for this stage, so the crop looks behind schedule.")
        elif s == "ahead":
            out.append(f"As of {case['as_of']}, {case['above']} of the last {case['checked']} ten-day periods in {g} were above the "
                       "expected greenness range; the crop looks ahead of schedule, which needs confirming (for example earlier planting).")
        else:
            out.append(f"As of {case['as_of']}, recent greenness in {g} is inside the range expected for this stage, so the crop looks on track.")
        out.append(f"In 30 days the crop is estimated to be at the {case['stage']} stage (estimated from the planting calendar, not observed). "
                   f"NDVI is expected to move from {case['last_ndvi']:.2f} to {case['f30']:.2f} (80% range {case['lo30']:.2f} to "
                   f"{case['hi30']:.2f}); the trend is {case['trend']}.")
    else:
        out.append(f"No satellite forecast exists for {g}; only yield history is available.")
    if "res_class" in case:
        out.append(f"Its resilience is in the {case['res_class']} third of the {len(D.ALL)} granaries: past shortfalls usually lasted about "
                   f"{case['typ_rec']} days. This is relative to the other granaries and has not been shown to predict yield.")
    if "yield_last" in case:
        t = f"Reported yield in {case['yield_year']} was {case['yield_last']:,.0f} kg/ha"
        if case.get("yield_change") is not None:
            t += f" ({case['yield_change']:+.1f}% on the previous year)"
        t += "."
        if "outlook" in case:
            t += (f" The outlook for {case['outlook_year']} is {case['outlook']:,.0f} kg/ha (range {case['outlook_low']:,.0f} to "
                  f"{case['outlook_high']:,.0f}); it is as accurate as 'same as last year', no better.")
        out.append(t)
    out.append(f"Confidence in this advice is {case['confidence']}: " + "; ".join(case["conf_reasons"]) + ".")
    return out


def memo_task(case, actions):
    guide = "\n".join(f"- {a[0]} (owner: {a[1]}; timing: {a[2]}; why: {a[3]})" for a in actions) or "- No rule matched."
    task = (f"Write a one-page decision memo for KPKM programme administrators about the granary '{case['granary']}'"
            + (f" (administrative area: {case['admin']})" if case["admin"] else "") + ". Sections: What is happening to the paddy; "
            "How confident we are; Recommended interventions (keep each owner and timing, add a 'Why:' line); What to verify "
            "before acting. Base the interventions ONLY on this approved guidance and the evidence:\n" + guide)
    return task
