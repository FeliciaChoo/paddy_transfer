"""PACE V8 governance views: one workspace per role, linked by results/governance_registry.json.

  Model Validator & Data Steward  writes the registry (tab 5: dual-criteria check, then Approve or Reject).
  KPKM Programme Manager          reads it (tabs 1, 3, 4, 5): the Approved Transfer forecast with a green badge, or the validated local baseline with an amber one.
                                  This role never runs, configures or sees a model parameter.
"""
import numpy as np
import pandas as pd
import streamlit as st

from acts_transfer import governance as gov
from acts_transfer.scenarios import SCENARIOS, scenario_shifts

from . import tab_overview
from .ui import selected_granary

ROLE_KPKM, ROLE_VAL = "KPKM Programme Manager", "Model Validator & Data Steward"
BASELINE_LABEL = {"season_trend": "Season trend", "persistence": "Persistence (last year's yield)", "target_mean": "Target mean", "target_only": "Target-only Ridge"}


def role():
    return st.session_state.get("workspace_role", ROLE_KPKM)


def _badge(eff):
    {"success": st.success, "warning": st.warning}[eff["kind"]](f"**{eff['badge']}**")


# ------------------------------------------------------------------------------------------------ local baselines for granaries with no sign-off
def local_backtest(yh_g: pd.DataFrame, min_train=3):
    """expanding-window back-test of the two yield-history baselines for one granary: season trend and persistence (past years only)"""
    h = yh_g.sort_values("Year")
    yr, y = h.Year.values.astype(float), h.Yield.values
    rows = []
    for k in range(1, len(h)):
        row = dict(season=int(yr[k]), actual=float(y[k]), persistence=float(y[k - 1]), season_trend=np.nan)
        if k >= min_train:
            row["season_trend"] = float(np.polyval(np.polyfit(yr[:k], y[:k], 1), yr[k]))
        rows.append(row)
    return pd.DataFrame(rows)


def local_outlook(yh_g: pd.DataFrame, seasons, min_train=3):
    """season-trend outlook with a band measured h seasons ahead on this granary's own history and widened with the horizon (never flat)"""
    h = yh_g.sort_values("Year")
    yr, y = h.Year.values.astype(float), h.Yield.values
    last = yr.max()
    c = np.polyfit(yr, y, 1) if len(h) >= 3 else (0.0, float(y.mean()))
    errs = {}
    for hz in range(1, 5):
        e = []
        for end in range(min_train, len(h)):
            if end - 1 + hz < len(h):
                e.append(abs(np.polyval(np.polyfit(yr[:end], y[:end], 1), yr[end - 1 + hz]) - y[end - 1 + hz]))
        if len(e) >= 2:
            errs[hz] = float(np.percentile(e, 80))
    rows = []
    for s in seasons:
        hz = int(round(s - last))
        b1 = errs.get(1, float(np.std(y)))
        band = b1 * np.sqrt(max(hz, 1))                                       # never narrower than the one-season error widened by sqrt(horizon)
        if errs:
            hh = min(hz, max(errs))
            band = max(band, errs[hh] * (np.sqrt(hz / hh) if hh != hz else 1.0))
        p = float(np.polyval(c, s))
        rows.append(dict(season=int(s), predicted=p, low=p - band, high=p + band))
    return rows


def _served(eff, scenario):
    """(rows, label, is_transfer) for the model the KPKM view serves for this granary under a scenario"""
    pub = eff["published"] or {}
    out = (pub.get("outlook") or {}).get(scenario) or {}
    if eff["serves_transfer"] and pub.get("transfer_model") in out:
        return out[pub["transfer_model"]], pub.get("transfer_label") or "Transfer model", True
    bm = pub.get("baseline_model")
    if bm in out:
        return out[bm], pub.get("baseline_label") or BASELINE_LABEL.get(bm, bm), False
    return None, None, False


# ------------------------------------------------------------------------------------------------ tab 1: overview and governance status (both roles)
def render_overview(D):
    reg = gov.load()
    st.subheader("Overview & Governance Status")
    n_app, n_loc, n_pend = gov.counts(D.ALL, reg)
    a, b, c, d = st.columns(4)
    a.metric("Granaries", len(D.ALL))
    b.metric("🟢 Approved Transfer", n_app, help="Granaries whose Detrended Residual Transfer forecast a Model Validator has signed off.")
    c.metric("🟠 Local Baseline", n_loc, help="Everything else is served by the validated season trend / persistence baseline.")
    d.metric("of which pending review", n_pend, help="No sign-off recorded yet: defaults safely to the local baseline.")
    if reg.get("error"):
        st.warning(reg["error"])
    rows = []
    for g in D.ALL:
        e = gov.effective(g, reg)
        ev = (e["entry"] or {}).get("evidence", {})
        rows.append({"Granary": g, "Status": e["badge"], "State": e["status"], "Model served": (e["published"].get("transfer_label") if e["serves_transfer"] else
                                                                                                 (e["published"].get("baseline_label") or "Season trend (local baseline)")) if e["entry"] else "Season trend (local baseline)",
                     "Delta_NT (kg/ha)": ev.get("delta_nt_kg_ha"), "Signed off by": (e["entry"] or {}).get("validator", ""), "Date": ((e["entry"] or {}).get("timestamp", "") or "")[:10]})
    with st.expander("Status of every granary", expanded=role() == ROLE_VAL):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if role() == ROLE_VAL:
        _registry_health(D, reg)
    tab_overview.render(D)


def _registry_health(D, reg):
    with st.container(border=True):
        st.markdown("**Dataset and governance registry health** (Model Validator & Data Steward)")
        files = {"Granary-year table with stage features (N = 70)": "transfer_stage_dataset.csv", "Ten-day driver table": "transfer_decade_features.csv",
                 "Benchmark sweep (Discover)": "benchmark_sweep_pooled.csv"}
        out = []
        for lab, f in files.items():
            p = D.DATA_DIR / f
            n = len(pd.read_csv(p, usecols=[0])) if p.exists() else None
            out.append({"Item": lab, "File": f, "Present": "✅" if p.exists() else "❌ missing", "Rows": n})
        st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
        ents = reg.get("granaries", {})
        last = max((e.get("timestamp", "") for e in ents.values()), default="")
        st.caption(f"Registry: {gov.DEFAULT_PATH} · {len(ents)} sign-off(s) recorded" + (f" · latest {last}" if last else "") + ". "
                   "A granary with no entry is PENDING_REVIEW and is served by the local baseline.")


# ------------------------------------------------------------------------------------------------ KPKM tab 3: historical replay
def render_replay(D):
    import plotly.graph_objects as go
    g = selected_granary()
    reg = gov.load()
    eff = gov.effective(g, reg)
    st.subheader("Historical Replay & Yield Fit")
    st.caption("Actual yield against what the forecast would have said, using only the years before each one. Nothing here needs a model setting.")
    _badge(eff)
    hist = D.yh[D.yh.District == g].sort_values("Year")
    pub = eff["published"] or {}
    bt = (pub.get("backtest") or {}).get("forward")
    if bt:
        df = pd.DataFrame(bt)
        served_col = "transfer" if eff["serves_transfer"] and df.get("transfer") is not None and df["transfer"].notna().any() else "baseline"
        served_name = pub.get("transfer_label") if served_col == "transfer" else pub.get("baseline_label")
        cmp_cols = {"persistence": "Persistence (last year)", "season_trend": "Season trend"}
    else:
        df = local_backtest(hist).rename(columns={"season_trend": "baseline"})
        df["season_trend"] = df["baseline"]
        served_col, served_name, cmp_cols = "baseline", "Season trend (local baseline)", {"persistence": "Persistence (last year)"}
    fig = go.Figure()
    fig.add_scatter(x=hist.Year, y=hist.Yield, mode="lines+markers", name="Reported yield", line=dict(color="#222", width=3))
    d = df.dropna(subset=[served_col])
    fig.add_scatter(x=d.season, y=d[served_col], mode="lines+markers", name=f"Replay: {served_name}", line=dict(color="#2ca02c" if served_col == "transfer" else "#ff7f0e", dash="dash"))
    for k, lab in cmp_cols.items():
        if k in df and df[k].notna().any() and k != served_col:
            fig.add_scatter(x=df.season, y=df[k], mode="lines", name=lab, line=dict(width=1, dash="dot"), opacity=.6)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="kg/ha", xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key="replay_chart")
    d = d.assign(error=d[served_col] - d.actual)
    m1, m2, m3 = st.columns(3)
    if len(d):
        m1.metric("Seasons replayed", len(d))
        m2.metric("Average miss (MAE)", f"{d.error.abs().mean():,.0f} kg/ha")
        m3.metric("Typical miss (RMSE)", f"{np.sqrt((d.error ** 2).mean()):,.0f} kg/ha", f"{100 * np.sqrt((d.error ** 2).mean()) / d.actual.mean():.1f}% of yield", delta_color="off")
    show = d.rename(columns={"season": "Season", "actual": "Actual (kg/ha)", served_col: "Replay forecast (kg/ha)", "error": "Forecast − actual (kg/ha)"})
    st.dataframe(show[["Season", "Actual (kg/ha)", "Replay forecast (kg/ha)", "Forecast − actual (kg/ha)"]].round(0), hide_index=True, width="stretch")
    st.caption("Replay uses only the past (expanding window). With so few seasons a miss of this size can happen by chance, so read it as the order of magnitude of the error, not a precise accuracy.")


# ------------------------------------------------------------------------------------------------ KPKM tab 4: nowcast and outlook
def render_outlook(D):
    import plotly.graph_objects as go
    g = selected_granary()
    reg = gov.load()
    eff = gov.effective(g, reg)
    pub = eff["published"] or {}
    st.subheader("In-Season Nowcast & 2025–2028 Outlook")
    _badge(eff)
    last_year = int(D.yh[D.yh.District == g].Year.max())
    st.markdown("### In-season nowcast (heading stage, about 70 days after sowing)")
    nc = pub.get("nowcast") or {}
    if nc:
        rowv = nc.get("transfer") if (eff["serves_transfer"] and nc.get("transfer")) else nc.get("baseline")
        who = pub.get("transfer_label") if (eff["serves_transfer"] and nc.get("transfer")) else pub.get("baseline_label")
        if rowv:
            a, b, c = st.columns(3)
            a.metric(f"Nowcast {nc['season']}", f"{rowv['predicted']:,.0f} kg/ha", f"{rowv['low']:,.0f} to {rowv['high']:,.0f}", delta_color="off")
            if nc.get("actual") is not None:
                b.metric("Reported yield", f"{nc['actual']:,.0f} kg/ha", f"{rowv['predicted'] - nc['actual']:+,.0f} vs nowcast", delta_color="off")
            c.metric("Model", who)
        st.caption(f"Uses the satellite stage features up to heading ({nc.get('stage', '')}) plus the trend. The latest season with such data is {nc['season']} (satellite data ends {D.AS_OF}), "
                   "so this is a back-test of the nowcast; a live nowcast for the current season needs that season's heading-stage data.")
    else:
        st.info("No validator-verified nowcast exists for this granary. Only the multi-year outlook below (season trend) is available.")

    st.markdown("### 2025–2028 outlook")
    hist = D.yh[D.yh.District == g].sort_values("Year")
    seasons = list(range(last_year + 1, last_year + 5))
    has_scen = bool(pub.get("outlook")) and eff["serves_transfer"]
    if has_scen:
        scen = st.radio("Scenario", SCENARIOS, horizontal=True, key="kp_scen",
                        help="Normal: typical stage conditions. El Niño drought: the dry, hot, bright, low-canopy tail of the year-to-year anomalies seen across the granaries in 2018 to 2024. "
                             "Wet monsoon: the wet tail. These are assumptions about what such a year looks like in this data, not weather forecasts.")
        rows, label, is_t = _served(eff, scen)
    else:
        scen = "Normal"
        rows, label, is_t = _served(eff, "Normal") if pub.get("outlook") else (None, None, False)
        if rows is None:
            rows, label = local_outlook(hist, seasons), "Season trend (local baseline)"
        if not eff["serves_transfer"]:
            st.caption("The local baseline is active, so the scenarios do not change this forecast (a season trend does not use climate inputs). "
                       "Scenarios become available once a Model Validator approves the transfer model for this granary.")
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=hist.Year, y=hist.Yield, mode="lines+markers", name="Reported yield", line=dict(color="#222", width=3))
    fig.add_scatter(x=df.season, y=df.high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=df.season, y=df.low, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(44,160,44,.18)" if is_t else "rgba(255,127,14,.18)", name="Uncertainty band")
    fig.add_scatter(x=df.season, y=df.predicted, mode="lines+markers", name=f"{label} · {scen}", line=dict(color="#2ca02c" if is_t else "#ff7f0e", width=3))
    if has_scen:
        for sc in SCENARIOS:
            if sc != scen:
                r2, _, _ = _served(eff, sc)
                if r2:
                    fig.add_scatter(x=[r["season"] for r in r2], y=[r["predicted"] for r in r2], mode="lines", name=sc, line=dict(width=1, dash="dot"), opacity=.7)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="kg/ha", xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key="outlook_chart")
    out = df.rename(columns={"season": "Season", "predicted": "Forecast (kg/ha)", "low": "Low (kg/ha)", "high": "High (kg/ha)"})
    out["± band"] = (df.high - df.low) / 2
    st.dataframe(out[["Season", "Forecast (kg/ha)", "Low (kg/ha)", "High (kg/ha)", "± band"]].round(0), hide_index=True, width="stretch")
    st.caption("The band widens with the horizon: it starts from the error measured on past held-out seasons and grows with the number of seasons ahead. It is an empirical range from a handful of "
               "seasons, not a guaranteed 80% interval.")
    if has_scen:
        sh = (pub.get("scenario_shifts") or {}).get(scen) or {}
        if sh:
            with st.expander("What this scenario changes"):
                st.dataframe(pd.DataFrame([{"Stage feature": k, "Shift from the typical value": round(v, 3)} for k, v in sh.items()]), hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------ validator tab 5: sign-off and assurance gate
def render_signoff(D):
    from acts_transfer.publish import build_published
    from . import tab_transfer as T
    st.subheader("Validator Sign-Off & Assurance Gate")
    wf = T._wf()
    reg = gov.load()
    st.caption("Dual-criteria check (Domain Relevance + Delta_NT < 0 against every local/trend baseline). Your decision is written to results/governance_registry.json and is what the KPKM Programme Manager sees.")
    if wf.choice is None or wf.target is None:
        st.info("Run the benchmark first: Tab 3 (audit) and Tab 4 (Train and Evaluate). The gate has nothing to check yet.")
        _history(reg)
        return
    c = wf.choice
    tgt = wf.target.name
    sg = c["safeguard"]
    cand = c["table"][c["table"]["deployable"]] if len(c["table"]) else c["table"]
    if cand.empty:
        st.warning("The Detrended Residual Transfer method was not part of this run, so there is nothing to approve: only methods that are deployable under the methodology can be approved. "
                   "Add it in the strategy sub-tab and train again, or record a rejection to keep the local baseline.")
        cur = gov.effective(wf.target.name, reg)
        st.markdown(f"**Current registry state for {wf.target.name}:** {cur['badge']} ({cur['status']})")
        who = st.text_input("Validator name", key="so_name")
        note = st.text_area("Sign-off note", key="so_note", height=70)
        if st.button("⛔ Reject to local baseline", key="so_reject"):
            _record(wf, D, wf.target.name, "REJECT", who, note, dict(gate_passed=False, model_key=None, model_label="none run", source=wf.source.name, token=c["token"]))
        _history(reg)
        return
    r = cand.sort_values("Delta_NT (RMSE)").iloc[0]
    kg = float(getattr(wf.bundle, "y_to_kg", 1.0))
    st.markdown(f"**Target:** {tgt} · **Source:** {wf.source.name} · **Method under review:** {T._lbl(r['method'])} · **Validation:** {T.primary_label(c)}")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Criterion 1 · Domain relevance**")
        st.markdown(("✅ Compatible agro-climatic calendar regime" if r["Criterion 1 domain relevance"] else "❌ Not compatible"))
        st.caption("The source pool contains no granary flagged by the divergence screen (KETARA, domain AUC 0.93 to 0.97, Northeast Monsoon calendar)." if r["Criterion 1 domain relevance"]
                   else (sg.get("domain_reason") or "A source granary is flagged by the divergence screen."))
    with b.container(border=True):
        st.markdown("**Criterion 2 · Baseline pass (Delta_NT < 0)**")
        st.markdown(("✅ " if r["Criterion 2 baseline pass"] else "❌ ") + f"Delta_NT = {r['Delta_NT (RMSE)'] * kg:+.0f} kg/ha against {T._lbl(r['Delta_NT baseline'])}")
        cr = r["RMSE vs all baselines (common seasons)"]
        rows = [{"Model": T._lbl(k), "RMSE (kg/ha)": v * kg, "": "← transfer" if k == r["method"] else ("← hardest local baseline" if k == r["Delta_NT baseline"] else "")} for k, v in cr.items()]
        st.dataframe(pd.DataFrame(rows).round(0), hide_index=True, width="stretch")
        st.caption(f"RMSE over the {int(r['seasons compared'])} held-out seasons common to the transfer model and all three baselines (the season trend needs 3 training years, so it is scored on fewer seasons).")
    ev_floor = r[f"at least {wf.run_cfg.min_test_seasons} common seasons"]
    gate = bool(r["gate passed"])
    if gate and not ev_floor:
        gate = False
    (st.success if gate else st.error)(("Gate PASSED: the transfer model may be approved." if gate else "Gate FAILED: the transfer model cannot be approved; the local baseline stays active."))
    if gate and not r["statistically supported"]:
        st.warning("The gate passed on the point estimate, but the evidence is weak: with this few held-out seasons the improvement is not statistically supported. Record that in your note.")
    cur = gov.effective(tgt, reg)
    st.markdown(f"**Current registry state for {tgt}:** {cur['badge']} ({cur['status']})" + (f" · signed off by {cur['entry']['validator']} on {cur['entry']['timestamp'][:10]}" if cur["entry"] else " · no sign-off recorded"))
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        who = c1.text_input("Validator name", key="so_name")
        note = c2.text_area("Sign-off note (evidence reviewed, caveats)", key="so_note", height=90)
        b1, b2, b3 = st.columns(3)
        evidence = dict(gate_passed=gate, model_key=r["method"], model_label=T._lbl(r["method"]), delta_nt=float(r["Delta_NT (RMSE)"]), delta_nt_kg_ha=float(r["Delta_NT (RMSE)"] * kg),
                        baseline=r["Delta_NT baseline"], domain_ok=bool(r["Criterion 1 domain relevance"]), seasons_compared=int(r["seasons compared"]), source=wf.source.name,
                        validation=T.primary_label(c), statistically_supported=bool(r["statistically supported"]), token=c["token"], rmse_common_kg_ha={k: float(v * kg) for k, v in cr.items()})
        if b1.button("✅ Approve transfer model", key="so_approve", disabled=not gate, type="primary"):
            _record(wf, D, tgt, "APPROVE", who, note, evidence)
        if b2.button("⛔ Reject to local baseline", key="so_reject"):
            _record(wf, D, tgt, "REJECT", who, note, evidence)
        if b3.button("↩ Reset to pending review", key="so_reset"):
            gov.clear(tgt)
            st.rerun()
    if st.session_state.pop("so_done", None):
        st.success(st.session_state.pop("so_msg", "Recorded."))
    _history(reg)


def _record(wf, D, tgt, decision, who, note, evidence):
    from acts_transfer.publish import build_published
    from . import tab_transfer as T
    if not (who or "").strip():
        st.error("Enter the validator's name: a sign-off is a named, recorded decision.")
        return
    got = T._data(D)
    try:
        pub = build_published(wf, got[0], tgt) if got else {}
        e = gov.record_signoff(tgt, decision, who, note, evidence, pub)
    except ValueError as ex:
        st.error(str(ex))
        return
    st.session_state["so_done"], st.session_state["so_msg"] = True, f"Recorded: {tgt} is now {e['status']}. The KPKM views will serve " + ("the verified transfer model." if e["status"] == gov.APPROVED else "the local baseline.")
    st.rerun()


def _history(reg):
    ents = reg.get("granaries", {})
    if ents:
        with st.expander("Registry contents"):
            st.dataframe(pd.DataFrame([{"Granary": k, "State": v["status"], "Validator": v.get("validator"), "Date": v.get("timestamp"), "Model": (v.get("evidence") or {}).get("model_label"),
                                        "Delta_NT (kg/ha)": (v.get("evidence") or {}).get("delta_nt_kg_ha"), "Note": v.get("note")} for k, v in ents.items()]), hide_index=True, width="stretch")
