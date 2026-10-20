"""Reliability and data: what the model can and cannot do. Short on purpose; the three main tabs are the product."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .ui import info_panel, selected_granary


def render(D):
    st.subheader("How reliable is this?")
    info_panel(
        "How well the NDVI forecast did on years it had not seen, how well simple yield baselines did, and the known limits "
        "of the data.",
        "Know how much weight to give an alert or a forecast: it is a prompt to check the field, not a diagnosis.",
        "Decide what can be used for planning now and what still needs data (validation events, season-level yield).",
        f"forecasts start from {D.AS_OF}; alerts and stage estimates have not been checked against field reports.")
    sk = D.skill.copy()
    sk["Horizon"] = sk.horizon.map(lambda x: f"{x * 10}-day")
    fig = go.Figure()
    for h_, g_ in sk.groupby("Horizon"):
        fig.add_trace(go.Bar(x=g_.test_year.astype(str), y=g_.skill_vs_persistence, name=h_))
    fig.update_layout(barmode="group", height=300, yaxis_title="Skill vs 'NDVI stays as it was'",
                      xaxis_title="Test year (trained on earlier years only)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sk[["Horizon", "test_year", "model", "MAE_model", "MAE_persistence", "skill_vs_persistence"]],
                 hide_index=True, width="stretch")

    st.markdown("**NDVI forecast against what was observed**")
    d = selected_granary()
    c1, c2 = st.columns(2)
    hz = c1.radio("Forecast horizon", [1, 3], format_func=lambda x: f"{x * 10} days", horizontal=True, key="rel_hz")
    yrs = sorted(int(y) for y in D.preds.test_year.unique())
    yr = c2.select_slider("Year", options=yrs, value=yrs[-1], key="rel_year")
    p = D.preds[(D.preds.District == d) & (D.preds.horizon == hz) & (D.preds.test_year == yr)].copy()
    if p.empty:
        st.info(f"No forecast data for {d} with this selection.")
    else:
        p["target_date"] = pd.to_datetime(p.target_date)
        out = (p.observed < p.lower) | (p.observed > p.upper)
        ok = p.dropna(subset=["lower"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(ok.target_date) + list(ok.target_date)[::-1], y=list(ok.upper) + list(ok.lower)[::-1],
                                 fill="toself", fillcolor="rgba(31,119,180,0.2)", line=dict(width=0), name="80% range",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p.target_date, y=p.forecast, mode="lines", name="Forecast", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=p.target_date[~out], y=p.observed[~out], mode="markers", name="Observed",
                                 marker=dict(color="black", size=6)))
        fig.add_trace(go.Scatter(x=p.target_date[out], y=p.observed[out], mode="markers", name="Observed, outside range",
                                 marker=dict(color="red", size=8)))
        fig.update_layout(height=320, yaxis_title="NDVI", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Red dots fall outside the forecast range: a possible behind- or ahead-of-schedule signal, not a confirmed problem.")

    st.markdown("**Yield methods, back-tested 1 to 4 years ahead (mean absolute % error; lower is better)**")
    bt = D.backtest.pivot(index="Method", columns="Horizon_years", values="MAPE_pct").round(1)
    bt.columns = [f"{c} year ahead" for c in bt.columns]
    st.dataframe(bt.reset_index(),
                 hide_index=True, width="stretch")
    weak = f"**{', '.join(D.WEAK)}** " + ("has" if len(D.WEAK) == 1 else "have") + " mixed planting dates, so stage estimates and forecasts are less reliable." if D.WEAK else ""
    st.markdown(f"""
**What this tool cannot do yet**
- **The yield outlook is transfer-learned from yield history only.** Carry-over rates are learned on the other granaries. It beats training on one granary alone, but it only matches 'same as last year': no satellite, weather or resilience input improved it with the yield data available.
- **Resilience is a relative index.** It is measured through the NDVI forecast (how often and how deeply greenness falls below the expected range, and how fast it recovers). Across the 10 granaries it showed no relationship with how stable the reported yield is.
- **Forecasts start from the end of the data ({D.AS_OF}).** Live use needs a regular update of satellite imagery.
- **Stage is estimated** from planting calendars inferred from NDVI, not from official dates.
- **Alerts are not validated** against field reports. A below-range decade is a prompt to check, not a diagnosis.
{('- ' + weak) if weak else ''}
- **Some yield values jump** between years and may reflect changes in how yield was counted.
- **Map areas** are the administrative districts of each granary (geo/granary_districts.csv); the satellite values come from sampled paddy pixels inside them, not from the whole district.
""")
