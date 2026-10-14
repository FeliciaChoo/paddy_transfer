"""Component 2 · Yield: past values on a map and graph, plus the future yield outlook, on one timeline."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .charts import BLUE, LIGHT_BLUE, yield_figure
from .mapkit import polygon_map, show_legend, view_for
from .ui import apply_map_click, info_panel, selected_granary

YIELD_SPEC = dict(col="Yield", fmt="{:,.0f}", unit=" kg/ha", kind="cont", low="lower yield", high="higher yield",
                  meaning="One colour scale for all years, so you can watch a granary change as you move the slider. "
                          "Years marked 'outlook' are projections, not measurements.")


def frame_for_year(D, year):
    """one row per granary for the chosen year: the reported value if there is one, otherwise the outlook value"""
    obs = D.yh[D.yh.Year == year][["District", "Yield"]].assign(Kind="observed", Low=np.nan, High=np.nan)
    fc = (D.outlook[D.outlook.Forecast_year == year][["District", "Central", "Low", "High"]]
          .rename(columns={"Central": "Yield"}).assign(Kind="outlook"))
    v = pd.concat([obs, fc], ignore_index=True).drop_duplicates("District", keep="first")
    m = D.fp.merge(D.districts[["District", "Latitude", "Longitude"]], on="District").merge(v, on="District", how="left")
    m["Note"] = [("" if pd.isna(k) else "observed" if k == "observed" else f"outlook, band {lo:,.0f} to {hi:,.0f}")
                 for k, lo, hi in zip(m.Kind, m.Low, m.High)]
    return m


def ranking_figure(frame):
    srt = frame.dropna(subset=["Yield"]).sort_values("Yield")
    colours = [BLUE if k == "observed" else LIGHT_BLUE for k in srt.Kind]
    fig = go.Figure(go.Bar(
        x=srt.Yield, y=srt.District, orientation="h", marker_color=colours,
        error_x=dict(type="data", symmetric=False, array=(srt.High - srt.Yield).fillna(0),
                     arrayminus=(srt.Yield - srt.Low).fillna(0), color="#999"),
        hovertemplate="%{y}: %{x:,.0f} kg/ha<extra></extra>"))
    fig.update_layout(height=max(260, 30 * len(srt) + 90), xaxis_title="Yield (kg/ha)  (grey bars = outlook band)",
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig


def render(D):
    st.subheader("Yield: past values and future outlook")
    info_panel(
        "Reported annual yield for every granary, then a future outlook for the next 1 to 4 years, on one timeline. Move the "
        "year slider from the past into the outlook years and the map, ranking and graph follow. The outlook is a baseline "
        "('same as last year' for the first two years, then the average of the last three years) with an uncertainty band "
        "taken from back-tests.",
        "Understand what yield is typical for your granary and how much it normally varies, so you can set realistic "
        "expectations and compare with your own field records.",
        "Use the outlook as a planning range for production, storage and inputs, and find granaries with unstable or "
        "suspicious records that must be checked at the source.",
        "the outlook is NOT a trained forecast. Satellite- and weather-based models did not beat 'same as last year' on this "
        "data. The band is wide on purpose.")
    sel = selected_granary()
    years = sorted(set(int(y) for y in D.yh.Year) | set(int(y) for y in D.outlook.Forecast_year))
    label = lambda y: f"{y} · outlook" if y > D.LAST_YEAR else str(y)  # noqa: E731
    c1, c2, c3 = st.columns([5, 1, 2])
    year = c1.select_slider("Year (past → outlook)", options=years, value=D.LAST_YEAR, format_func=label, key="yield_year")
    show_names = c2.checkbox("Names", value=True, key="yield_names")
    zoom_to = c3.radio("Zoom to", ["All granaries", "Selected granary"], horizontal=True, key="yield_zoom")
    is_outlook = year > D.LAST_YEAR
    frame = frame_for_year(D, year)
    v = frame.dropna(subset=["Yield"]).set_index("District").Yield

    # ---- headline numbers for the chosen year
    if is_outlook:
        ref = D.outlook.groupby("District").Last_yield.first()
        ref_name = "latest observed"
    else:
        ref = D.yh[D.yh.Year == year - 1].set_index("District").Yield
        ref_name = str(year - 1)
    common = v.index.intersection(ref.index)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Average yield, {year}" + (" (outlook)" if is_outlook else ""), f"{v.mean():,.0f} kg/ha" if len(v) else "n/a",
              help="Simple average over granaries; planted area is not available, so it is not area-weighted.")
    if len(common):
        k2.metric(f"Change vs {ref_name}", f"{(v[common].mean() / ref[common].mean() - 1) * 100:+.1f}%")
        k3.metric("Granaries higher", f"{int((v[common] > ref[common] * 1.005).sum())} / {len(common)}")
        k4.metric("Granaries lower", f"{int((v[common] < ref[common] * 0.995).sum())} / {len(common)}")
    else:
        k2.metric(f"Change vs {ref_name}", "n/a")
    if is_outlook:
        o_t = D.outlook[D.outlook.Forecast_year == year]
        st.caption(f"Outlook method for {year}: {o_t.Method.iloc[0]}. Typical error band: ±{o_t.Band_pct.iloc[0]:.0f}% "
                   "(80th percentile of back-test errors). 'Higher / lower' counts stay at zero while the method is 'same as "
                   "last year'.")

    # ---- map + ranking
    mcol, lcol = st.columns([3, 1])
    vmin = min(D.yh.Yield.min(), D.outlook.Low.min())
    vmax = max(D.yh.Yield.max(), D.outlook.High.max())
    with mcol:
        view = view_for(D, sel if zoom_to == "Selected granary" else None)
        lo, hi, clicked, present = polygon_map(frame, YIELD_SPEC, view, selected=sel, show_names=show_names,
                                               key="map_yield", vmin=vmin, vmax=vmax)
    with lcol:
        show_legend(YIELD_SPEC, lo, hi, present)
    apply_map_click(clicked, "yield")
    st.plotly_chart(ranking_figure(frame), width="stretch")

    # ---- one granary: past + outlook graph
    st.markdown(f"### {sel}: past yield and outlook")
    compare = st.multiselect("Compare with other granaries (reported yield)", [x for x in D.ALL if x != sel], key="yield_compare")
    st.plotly_chart(yield_figure(D, sel, height=400, compare=compare, marker_year=year), width="stretch")
    g = D.yh[D.yh.District == sel].sort_values("Year")
    o = D.outlook[D.outlook.District == sel].sort_values("Forecast_year")
    b1, b2, b3 = st.columns(3)
    if len(o):
        far = o.iloc[-1]
        b1.metric(f"Outlook {int(far.Forecast_year)}", f"{far.Central:,.0f} kg/ha", f"band {far.Low:,.0f} to {far.High:,.0f}",
                  delta_color="off")
        b2.metric("Change from latest reported", f"{(far.Central / g.Yield.iloc[-1] - 1) * 100:+.1f}%")
        b3.metric("Outlook horizon", f"{int(far.Horizon_years)} years")
    shared = g.Shares_series_with.iloc[0] if len(g) else ""
    if isinstance(shared, str) and shared:
        st.warning(f"{sel} reports the same yield series as: {shared}. They are not independent measurements.")
    if g.Flag_large_jump.any():
        st.warning("Large year-to-year jumps in: " + ", ".join(map(str, g[g.Flag_large_jump].Year)) +
                   ". These may be changes in how yield was counted, not crop responses; check the source.")
    st.info("These are exploratory outlooks, not guaranteed outcomes: climate, management, policy and cropping conditions "
            "can change actual yield.")

    with st.expander("Table: every granary, past and outlook (kg/ha)"):
        obs = D.yh.pivot(index="District", columns="Year", values="Yield")
        obs.columns = [str(c) for c in obs.columns]
        fc = D.outlook.pivot(index="District", columns="Forecast_year", values="Central")
        fc.columns = [f"{c} (outlook)" for c in fc.columns]
        st.dataframe(obs.join(fc).round(0).reset_index(), hide_index=True, width="stretch")
    with st.expander("How the outlook is built"):
        how = D.outlook.groupby("Horizon_years").agg(Method=("Method", "first"), Band_pct=("Band_pct", "first")).reset_index()
        st.dataframe(how, hide_index=True, width="stretch")
        st.markdown("Source file: `data/yield_outlook.csv`. To show a trained model instead, write its predictions to the same "
                    "columns (District, Last_year, Last_yield, Forecast_year, Horizon_years, Central, Band_pct, Low, High, "
                    "Method); this tab, the maps and the advisory tab pick them up without code changes.")
