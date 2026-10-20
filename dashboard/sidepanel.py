"""The panel BESIDE a map: the legend, then everything needed about the granary that is selected (clicked), so nobody scrolls.

It sits in a fixed-height box as tall as the map; if the content is longer than the map, the box scrolls on its own and the page does not.
"""
import pandas as pd
import streamlit as st

from . import bridge
from .charts import ndvi_figure, yield_figure
from .mapkit import show_legend

STATUS_ICON = {"behind": "🔴", "on track": "🟢", "ahead": "🔵"}


def side_panel(D, g, spec, lo, hi, present, height, tag, chart="yield", layer_line=None):
    """legend + the selected granary's key facts, a small chart and the most urgent intervention.

    chart: 'yield' (past yield and outlook), 'ndvi' (history and 30-day forecast) or None.
    layer_line: one line saying what the map layer shows for this granary (for example the value being coloured).
    tag: any text that is unique to the tab calling this (it keeps the chart widgets apart)."""
    case = bridge.build_case(D, g)
    acts = bridge.interventions(case, D.irules)
    with st.container(height=height, border=True):
        show_legend(spec, lo, hi, present, key=f"legend_{tag}")
        st.markdown(f"### {g}")
        if case["admin"]:
            st.caption(f"{case['admin']}  ·  click another granary on the map to change")
        else:
            st.caption("Click another granary on the map to change")
        if layer_line:
            st.markdown(layer_line)
        if case["has_sat"]:
            st.markdown(f"{STATUS_ICON[case['status']]} **{case['status'].upper()}**: {case['below']} of {case['checked']} recent "
                        f"periods below the expected range")
            a, b = st.columns(2)
            a.metric("Stage in 30 days", case["stage"].split(" / ")[0], help="Estimated from the planting calendar, not observed.")
            b.metric("NDVI trend", case["trend"], f"{case['last_ndvi']:.2f} to {case['f30']:.2f}", delta_color="off")
            c, d = st.columns(2)
            c.metric("Confidence", case["confidence"].upper(), help="; ".join(case["conf_reasons"]))
            if "res_class" in case:
                d.metric("Resilience", case["res_class"].upper(), f"index {case['res_index']:.0f} / 100", delta_color="off",
                         help="Measured through the NDVI forecast; relative to the other granaries.")
        else:
            st.info("No satellite forecast for this granary; only yield history exists.")
        if "yield_last" in case:
            line = f"**Yield {case['yield_year']}:** {case['yield_last']:,.0f} kg/ha"
            if "outlook" in case:
                line += (f"  \n**Outlook {case['outlook_year']}:** {case['outlook']:,.0f} kg/ha "
                         f"({case['outlook_low']:,.0f} to {case['outlook_high']:,.0f})")
            st.markdown(line)
        if chart == "yield":
            st.plotly_chart(yield_figure(D, g, height=210), width="stretch", key=f"sp_y_{tag}_{g}")
        elif chart == "ndvi":
            fig = ndvi_figure(D, g, height=210)
            if fig is None:
                st.caption("No NDVI history for this granary.")
            else:
                st.plotly_chart(fig, width="stretch", key=f"sp_n_{tag}_{g}")
        if acts:
            a0 = acts[0]
            st.markdown(f"**Most urgent ({bridge.priority_label(a0[7])}):** {a0[0]}")
            st.caption(f"Who: {a0[1]}  ·  When: {a0[2]}.  Full advice and the report: tab 4, Advisory.")
        if case.get("has_sat") and case["stale_days"] > 45:
            st.caption(f"Satellite data ends {case['as_of']} ({case['stale_days']} days ago).")


def yield_panel(D, g, spec, lo, hi, present, height, mode, year):
    """legend + the selected granary's YIELD facts only (reported yield, or the outlook). No crop status, advice or other details.

    mode 'past'    : yield reported for `year`, change on the previous year, position against the granary's own average, rank.
    mode 'outlook' : the outlook for `year`, its range and typical error, and how it compares with the latest reported yield."""
    from . import adminmap
    pts = adminmap.load()
    admin = pts[1].set_index("District").Admin.get(g, "") if pts is not None else ""
    with st.container(height=height, border=True):
        show_legend(spec, lo, hi, present, key=f"legend_yield_{mode}")
        st.markdown(f"### {g}")
        st.caption((admin + "  ·  " if admin else "") + "click another granary on the map to change")
        if mode == "past":
            gy = D.yh[D.yh.District == g].set_index("Year")
            if year in gy.index:
                r = gy.loc[year]
                every = D.yh[D.yh.Year == year].Yield
                rank = int((every > r.Yield).sum()) + 1
                a, b = st.columns(2)
                a.metric(f"Yield {year}", f"{r.Yield:,.0f} kg/ha")
                b.metric("Change on previous year", "n/a" if pd.isna(r.Change_pct) else f"{r.Change_pct:+.1f}%")
                c, d = st.columns(2)
                c.metric("Against its own average", f"{r.Yield_z:+.1f} SD", help="0 means a typical year for this granary; +1 is clearly above it.")
                d.metric("Rank this year", f"{rank} of {len(every)}")
                if g in D.MEAN_YIELD.index:
                    st.caption(f"Average of all reported years: {D.MEAN_YIELD[g]:,.0f} kg/ha.")
            else:
                st.info(f"No reported yield for {g} in {year}.")
        else:
            o = D.outlook[(D.outlook.District == g) & (D.outlook.Forecast_year == year)]
            if len(o):
                r = o.iloc[0]
                a, b = st.columns(2)
                a.metric(f"Outlook {year}", f"{r.Central:,.0f} kg/ha")
                b.metric("Typical error", f"±{r.Band_pct:.0f}%", help="80th percentile of the errors this method made in back-tests.")
                c, d = st.columns(2)
                c.metric("Range", f"{r.Low:,.0f} to {r.High:,.0f}")
                d.metric(f"Latest reported ({int(r.Last_year)})", f"{r.Last_yield:,.0f}", f"{(r.Central / r.Last_yield - 1) * 100:+.1f}% expected",
                         delta_color="off")
                st.caption(str(r.Method))
            else:
                st.info(f"No outlook for {g} in {year}.")
        gy = D.yh[D.yh.District == g]
        if len(gy) and isinstance(gy.Shares_series_with.iloc[0], str) and gy.Shares_series_with.iloc[0]:
            st.caption(f"Note: {g} reports the same yield figures as {gy.Shares_series_with.iloc[0]}.")
