"""Component 1 · Dynamic maps: pick a layer (and year), see every granary coloured, click one to open its profile."""
import pandas as pd
import streamlit as st

from .charts import ndvi_figure, yield_figure
from .mapkit import polygon_map, show_legend, view_for
from .ui import apply_map_click, info_panel, selected_granary

STATUS_COLOURS = {"behind": (214, 39, 40), "on track": (44, 160, 44), "ahead": (31, 119, 180)}
STAGE_COLOURS = {"germination / establishment": (199, 233, 180), "seedling": (127, 205, 187), "tillering": (65, 182, 196),
                 "max tillering / stem elongation": (29, 145, 192), "panicle initiation / booting": (34, 94, 168),
                 "heading / flowering": (254, 196, 79), "grain filling": (236, 112, 20),
                 "maturity / harvest": (153, 52, 4), "fallow / turnaround": (160, 160, 160)}


def layer_specs(final_year):
    """layer name -> how to colour it. 'year': the layer changes with the year slider."""
    past = {
        "Reported yield (kg/ha)": dict(
            col="Yield", fmt="{:,.0f}", unit=" kg/ha", year=True, kind="cont", low="lower yield", high="higher yield",
            meaning="Compare granaries in the same year. A high value is not automatically 'good': these are annual "
                    "granary-level figures and some granaries share one reported series."),
        "Yield versus the granary's own history (SD)": dict(
            col="Yield_z", fmt="{:+.2f}", year=True, kind="cont", low="lower than this granary's usual",
            high="higher than this granary's usual",
            meaning="0 = a typical year for that granary. +1 means one standard deviation above its own average year."),
        "Yield change versus previous year (%)": dict(
            col="Change_pct", fmt="{:+.1f}", year=True, kind="cont", low="yield fell", high="yield rose",
            meaning="Jumps of 20% or more may be changes in how yield was counted, not crop responses."),
    }
    now = {
        "Recent crop status": dict(
            col="Recent_status", fmt="{}", year=False, kind="cat", colours=STATUS_COLOURS,
            meaning="Behind = recent NDVI fell below the forecast range in at least 2 of the last 4 decades (a candidate "
                    "for a field check, not a confirmed problem). Ahead = above the range. On track = within it."),
        "Last NDVI (end of data)": dict(
            col="Last_NDVI", fmt="{:.2f}", year=False, kind="cont", low="sparse canopy, flooded field or early growth",
            high="dense green canopy",
            meaning="NDVI is satellite greenness (0 to 1). It depends on the crop stage, so read it with the stage."),
        "Expected NDVI change over the next 30 days": dict(
            col="Expected_change_30d", fmt="{:+.2f}", year=False, kind="cont",
            low="expected to fall (maturing or between crops)", high="expected to get greener (canopy developing)",
            meaning="Forecast 30-day NDVI minus the last observed NDVI."),
        "Estimated growth stage in 30 days": dict(
            col="Stage_30d", fmt="{}", year=False, kind="cat", colours=STAGE_COLOURS,
            meaning="Estimated from each granary's planting calendar, not observed in the field."),
        f"Share of {final_year} decades below the forecast range (10-day)": dict(
            col="Below_share_10d", fmt="{:.0%}", year=False, kind="cont", low="rarely below what was expected",
            high="often below what was expected (more to check)",
            meaning="About 10% is expected by chance from an 80% forecast range."),
        "Estimated planting decade (1 to 36)": dict(
            col="trough_A", fmt="{:.0f}", year=False, kind="cont", low="early in the year (January)",
            high="late in the year (December)",
            meaning="Decade 1 = early January, decade 36 = mid December. Estimated from NDVI, not from official calendars."),
        "NDVI seasonal amplitude": dict(
            col="NDVI_amplitude", fmt="{:.2f}", year=False, kind="cont", low="weak cycle or mixed planting dates",
            high="strong, regular cropping cycle",
            meaning="Peak minus trough of the typical year. Weak values make stage estimates less reliable."),
    }
    return past, now


def profile_card(D, d):
    """everything the app knows about one granary"""
    st.markdown(f"### Granary profile: {d}")
    g = D.yh[D.yh.District == d].sort_values("Year")
    o = D.outlook[D.outlook.District == d].sort_values("Horizon_years")
    s = D.cond[D.cond.District == d]
    s = s.iloc[0] if len(s) else None
    ph = D.phen[D.phen.District == d]
    ph = ph.iloc[0] if len(ph) else None
    last = g.iloc[-1]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Yield**")
        st.metric(f"Reported yield {int(last.Year)}", f"{last.Yield:,.0f} kg/ha",
                  None if pd.isna(last.Change_pct) else f"{last.Change_pct:+.1f}% vs previous year")
        st.metric("Against its own history", f"{last.Yield_z:+.2f} SD")
        if len(o):
            nxt = o.iloc[0]
            st.metric(f"Outlook {int(nxt.Forecast_year)}", f"{nxt.Central:,.0f} kg/ha",
                      f"band {nxt.Low:,.0f} to {nxt.High:,.0f}", delta_color="off")
    with c2:
        st.markdown("**Crop condition (NDVI)**")
        if s is None:
            st.info("No satellite forecast exists for this granary yet.")
        else:
            st.metric("Last NDVI", f"{s.Last_NDVI:.2f}", f"as of {s.As_of}", delta_color="off")
            st.metric("30-day forecast", f"{s.Forecast_30d:.2f}", f"{s.Expected_change_30d:+.2f}")
            st.metric("Recent status", s.Recent_status.upper(),
                      f"{int(s.Recent_below)} of {int(s.Recent_checked)} recent decades below range", delta_color="off")
    with c3:
        st.markdown("**Season**")
        if ph is None:
            st.info("No planting calendar estimate for this granary.")
        else:
            st.metric("Planting (estimated)", f"{ph.trough_A_date} and {ph.trough_B_date}",
                      f"peaks {ph.peak_A_date} and {ph.peak_B_date}", delta_color="off")
            st.metric("Seasonal signal", f"{ph.NDVI_amplitude:.2f}", str(ph.confidence), delta_color="off")
        if s is not None:
            st.metric("Stage in 30 days (estimated)", s.Stage_30d.split(" / ")[0])
    if g.Flag_large_jump.any():
        st.warning("Large year-to-year yield jumps in " + ", ".join(map(str, g[g.Flag_large_jump].Year)) +
                   " (may be counting changes).")
    if isinstance(last.Shares_series_with, str) and last.Shares_series_with:
        st.warning(f"Reports the same yield series as {last.Shares_series_with}.")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Yield: past and outlook**")
        st.plotly_chart(yield_figure(D, d, height=300), width="stretch")
    with ch2:
        st.markdown("**NDVI: history and 30-day forecast**")
        fig = ndvi_figure(D, d, height=300)
        if fig is None:
            st.info("No NDVI history for this granary in the prepared data.")
        else:
            st.plotly_chart(fig, width="stretch")


def render(D):
    st.subheader("Dynamic maps")
    info_panel(
        "Each granary is coloured by the value you choose. Yield layers use reported annual yield "
        f"({len(D.ALL)} granaries) for the year on the slider. Crop-condition layers use the latest satellite forecast "
        f"(as of {D.AS_OF}, {len(D.SAT)} granaries); pale means no value. Click a granary (or pick it in the sidebar) to "
        "open its profile.",
        "See how your granary compares with its neighbours and with its own past, and whether the current crop condition "
        "looks unusual, before you talk to the extension officer.",
        "See at a glance which granaries are high, low or unusual and decide where to look first.",
        "yields are annual granary-level figures and some granaries share one series; satellite layers exist for "
        f"{len(D.SAT)} of {len(D.ALL)} granaries. Each coloured area is the administrative district(s) the granary covers (not the exact paddy fields).")
    sel = selected_granary()
    past, now = layer_specs(D.FINAL_TEST_YEAR)

    a, b = st.columns([1, 2])
    group = a.radio("Show", ["Yield (past)", "Crop condition and season"], key="map_group")
    specs = past if group.startswith("Yield") else now
    layer = b.selectbox("Map layer", list(specs), key="map_layer_" + group[:4])
    spec = specs[layer]
    years = sorted(int(y) for y in D.yh.Year.unique())
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    yr = c1.select_slider("Year", options=years, value=years[-1], disabled=not spec["year"], key="map_year")
    zoom_to = c2.radio("Zoom to", ["All granaries", "Selected granary"], horizontal=True, key="map_zoom")
    show_names = c3.checkbox("Names", value=True, key="map_names")
    show_dots = c4.checkbox("Dots", value=False, key="map_dots",
                            help="A dot at each granary centre so small granaries stay visible when zoomed out.")

    m = (D.fp.merge(D.districts[["District", "Latitude", "Longitude"]], on="District")
         .merge(D.cond, on="District", how="left")
         .merge(D.phen[["District", "trough_A"]], on="District", how="left")
         .merge(D.yh[D.yh.Year == yr][["District", "Yield", "Yield_z", "Change_pct"]], on="District", how="left"))
    view = view_for(D, sel if zoom_to == "Selected granary" else None)
    mcol, lcol = st.columns([3, 1])
    with mcol:
        lo, hi, clicked, present = polygon_map(m, spec, view, selected=sel, show_dots=show_dots, show_names=show_names,
                                               key="map_main")
    with lcol:
        show_legend(spec, lo, hi, present)
    apply_map_click(clicked, "map")

    profile_card(D, sel)

    col = spec["col"]
    st.markdown(f"**Values behind the map: {layer}{' (' + str(yr) + ')' if spec['year'] else ''}**")
    extra = [c for c in ["Last_NDVI", "Stage_30d", "Recent_status"] if c != col]
    st.dataframe(m[["District", col] + extra].sort_values(col, ascending=False), hide_index=True, width="stretch")
