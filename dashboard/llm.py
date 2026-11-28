"""AI memo (Google Gemini) with guard-rails, run entirely on the server.

The API key is never typed into the app. It is read, in this order, from
  1. the environment variable GEMINI_API_KEY (or GOOGLE_API_KEY), or
  2. .streamlit/secrets.toml  ->  GEMINI_API_KEY = "..."   (on Streamlit Cloud: App settings -> Secrets).
Without a key the app still works: the interventions come from the editable rules and only the AI memo is skipped.
"""
import json
import os
import re
import time

import requests
import streamlit as st

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

SYSTEM = (
    "You write plain-English decision briefs for KPKM (Ministry of Agriculture and Food Security) programme managers "
    "overseeing paddy (rice) granaries in Malaysia. Use ONLY the numbered evidence items provided and cite them like [E3]. "
    "Never invent numbers, chemicals, doses, causes, or dates. Satellite imagery cannot prove a field cause on its own, "
    "so frame field actions as checks/verifications using conditional phrasing ('may', 'could'). Keep sentences short, "
    "clear, and completely free of technical jargon."
)


def _cfg(*names):
    for n in names:
        if os.environ.get(n):
            return os.environ[n].strip()
    for n in names:
        try:
            v = st.secrets.get(n, "")
        except Exception:
            v = ""
        if v:
            return str(v).strip()
    return ""


def api_key():
    return _cfg("GEMINI_API_KEY", "GOOGLE_API_KEY")


def llm_available():
    return bool(api_key())


def call_llm(system, user):
    models = [m.strip() for m in _cfg("GEMINI_MODEL").split(",") if m.strip()] or DEFAULT_MODELS
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2000},
    }
    for m in models:
        r = requests.post(
            f"{GEMINI_BASE}{m}:generateContent",
            headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if r.status_code == 404:
            continue
        r.raise_for_status()
        cand = (r.json().get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", [])).strip()
        if not text:
            raise ValueError(f"empty response (finishReason={cand.get('finishReason')})")
        return text
    raise RuntimeError(f"none of the Gemini models {models} was found; set GEMINI_MODEL to an available model")


def unverified_numbers(text, evidence_text):
    t = re.sub(r"\[E\d+(?:\s*,\s*E\d+)*\]", " ", text)
    t = re.sub(r"(?m)^\s*\d+[.)]\s", " ", t)
    ev_nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", evidence_text)]
    bad = []
    for s in re.findall(r"\d[\d,]*\.?\d*", t):
        try:
            v = float(s.replace(",", ""))
        except ValueError:
            continue
        if not any(abs(v - e) <= max(0.01 * abs(e), 0.006) for e in ev_nums):
            bad.append(s)
    return sorted(set(bad))


def run_llm(task, evidence):
    ev = "\n".join(f"[{i}] {t}" for i, t in evidence)
    out = call_llm(SYSTEM, f"{task}\n\nEVIDENCE:\n{ev}")
    return out, unverified_numbers(out, ev)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _memo(task, evidence):
    return run_llm(task, list(evidence))


def memo(task, evidence):
    if not llm_available():
        return None, [], "not_configured"
    if time.time() < st.session_state.get("_ai_cooldown", 0):
        return None, [], "temporarily_unavailable"
    try:
        text, bad = _memo(task, tuple(evidence))
        return text, bad, None
    except Exception as e:
        st.session_state["_ai_cooldown"] = time.time() + 120
        return None, [], type(e).__name__


def generate_manager_insight(
    granary: str,
    case: dict,
    acts: list,
    schedule_info: str,
    matched_refs: list[dict],
):
    """Generate a structured, jargon-free executive brief grounded in case metrics, irrigation schedules, and matched SOPs."""
    evidence = [
        ("E1", f"Granary={granary}; Stage={case.get('stage', 'unknown')}; Crop_Condition={case.get('status', 'unknown')}"),
        ("E2", f"Satellite_Available={case.get('has_sat')}; Below_Range_Periods={case.get('below', 0)} of {case.get('checked', 0)}"),
    ]
    if case.get("pred_line"):
        evidence.append(("E3", f"Yield_Forecast={case['pred_line']}"))
    if case.get("res_class"):
        evidence.append(("E4", f"Granary_Resilience={case['res_class']} (score {case.get('res_index', 0):.0f}/100)"))
    if schedule_info:
        evidence.append(("E5", f"Official_Irrigation_Schedule={schedule_info}"))

    for idx, a in enumerate(acts[:3], start=6):
        evidence.append((f"E{idx}", f"Priority_Action={a[0]}; Responsible_Officer={a[1]}; Timing={a[2]}; Reason={a[3]}"))
    for idx, r in enumerate(matched_refs, start=10):
        evidence.append((f"E{idx}", f"Validated_Manual={r['Source']}; Guidance={r['What']} ({r['Window']})"))

    task = (
        f"Create a clear, executive-friendly decision brief for a Programme Manager overseeing {granary}. "
        "Do NOT use technical jargon (avoid words like NDVI, HST, LSWI, decade, indicative anchor, or sample coverage). "
        "Translate everything into everyday operational language.\n"
        "Return ONLY valid JSON with these exact 7 keys (no markdown code blocks):\n"
        "{\n"
        '  "status_label": "Action Needed" or "Review Recommended" or "On Track",\n'
        '  "headline": "Clear 6-10 word action title",\n'
        '  "situation": "2 short plain sentences summarizing crop status, schedule phase, and what needs attention.",\n'
        '  "action_step": "1-2 clear sentences stating the main field verification step.",\n'
        '  "lead_officer": "Primary role responsible (e.g., Extension Agronomist & Block Coordinator)",\n'
        '  "when": "Clear timing phrase (e.g., During this week\'s field round)",\n'
        '  "purpose": "Clear operational goal (e.g., Confirm local planting dates and canal water delivery before input changes)"\n'
        "}"
    )
    raw, bad, err = memo(task, evidence)
    if not raw:
        return None, bad, err
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned), bad, None
    except Exception:
        return None, bad, "parse_error"


def narrate_counterfactual_cards(granary: str, prescriptions: list[dict]):
    if not prescriptions:
        return None, [], "no_prescriptions"
    evidence = []
    for idx, p in enumerate(prescriptions, start=1):
        evidence.append((
            f"E{idx}",
            f"Granary={granary}; Lever={p.get('lever')}; Action={p.get('action')}; "
            f"Target_Window={p.get('window')}; Expected_Yield_Gain_kg_ha={p.get('delta_y_kg_ha')}; "
            f"Confidence_Interval_kg_ha=({p.get('ci_low')}, {p.get('ci_high')}); "
            f"Schedule_Bound={p.get('schedule_bound')}; SOP_Rule={p.get('rule_id', 'N/A')}"
        ))
    task = (
        f"Translate the verified counterfactual levers for {granary} into concise field-officer advisory cards. "
        "For each lever, format strictly with four bold labels: "
        "**Who** (responsible agency/block officers), **What** (concrete management check or schedule adjustment), "
        "**When** (exact decade/date window from evidence), and **Why** (cited expected yield gain in kg/ha and "
        "agronomic mechanism from evidence). Cite every figure like [E1]."
    )
    return memo(task, evidence)