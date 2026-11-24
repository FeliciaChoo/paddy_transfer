"""AI memo (Google Gemini) with guard-rails, run entirely on the server.

The API key is never typed into the app. It is read, in this order, from
  1. the environment variable GEMINI_API_KEY (or GOOGLE_API_KEY), or
  2. .streamlit/secrets.toml  ->  GEMINI_API_KEY = "..."   (on Streamlit Cloud: App settings -> Secrets).
Without a key the app still works: the interventions come from the editable rules and only the AI memo is skipped.
"""
import os
import re
import time

import requests
import streamlit as st

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]        # override with env GEMINI_MODEL="a,b" (tried in order)

SYSTEM = (
    "You write decision-support memos for KPKM (Ministry of Agriculture and Food Security) programme administrators "
    "and IADA/MADA/KADA field officers about paddy (rice) granaries in Malaysia. Use ONLY the numbered evidence items "
    "you are given and cite them like [E3]. Do not invent numbers, causes, doses, chemicals or dates. Satellite "
    "greenness (NDVI) cannot diagnose a cause, so recommend things to CHECK in the field before any intervention, and "
    "use conditional wording ('may', 'could'). Be concise and plain. If evidence is weak or missing, say so. "
    "This is not a field diagnosis."
)


def _cfg(*names):
    """first non-empty value among environment variables, then .streamlit/secrets.toml"""
    for n in names:
        if os.environ.get(n):
            return os.environ[n].strip()
    for n in names:
        try:
            v = st.secrets.get(n, "")
        except Exception:                                  # no secrets file: that is fine
            v = ""
        if v:
            return str(v).strip()
    return ""


def api_key():
    return _cfg("GEMINI_API_KEY", "GOOGLE_API_KEY")


def llm_available():
    return bool(api_key())


def call_llm(system, user):
    """Gemini generateContent (REST). Tries each model in turn if one is not found (HTTP 404)."""
    models = [m.strip() for m in _cfg("GEMINI_MODEL").split(",") if m.strip()] or DEFAULT_MODELS
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2000}}
    for m in models:
        r = requests.post(f"{GEMINI_BASE}{m}:generateContent",
                          headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"}, json=body, timeout=60)
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
    """numbers in the AI text that do not appear (within rounding) in the evidence items"""
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
def _memo(task, evidence):                                  # same task + evidence = one API call, then served from cache
    return run_llm(task, list(evidence))


def memo(task, evidence):
    """(text, numbers_not_in_evidence, problem). Generated on the server; never raises, so a failure cannot break the page."""
    if not llm_available():
        return None, [], "not_configured"
    if time.time() < st.session_state.get("_ai_cooldown", 0):          # after a failure, do not retry on every click
        return None, [], "temporarily_unavailable"
    try:
        text, bad = _memo(task, tuple(evidence))
        return text, bad, None
    except Exception as e:                                              # network, key or model problems
        st.session_state["_ai_cooldown"] = time.time() + 120
        return None, [], type(e).__name__

def narrate_counterfactual_cards(granary: str, prescriptions: list[dict]):
    """Render schedule-bounded causal prescriptions into Who / What / When / Why briefs."""
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