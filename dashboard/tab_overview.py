"""Component 1 · Overview: the whole picture on one page -- headline numbers, a status map, a granary scoreboard, data notes."""
import numpy as np
import pandas as pd
import streamlit as st

from . import tab_reliability
from .mapkit import polygon_map, show_legend, view_for
from .tab_map import STATUS_COLOURS
from .ui import apply_map_click, info_panel, selected_granary

STATUS_SPEC = dict(col="Recent_status", fmt="{}", kind="cat", colours=STATUS_COLOURS,
                   meaning="Behind = recent NDVI fell below the forecast range in at least 2 of the last 4 decades (a candidate "
                           "for a field check, not a confirmed problem). Ahead = above the range. On track = within it.")


def scoreboard(D):
    """one row per granary: latest yield, how it moved, current crop condition, next-year outlook"""
    last = D.yh[D.yh.Year == D.LAST_YEAR].set_index("District")
    o1 = D.outlook[D.outlook.Horizon_years == D.outlook.Horizon_years.min()].set_index("District")
    t = pd.DataFrame(index=D.ALL)
    t[f"Yield {D.LAST_YEAR} (kg/ha)"] = last.Yield
    t["Change vs previous year (%)"] = last.Change_pct
    t["vs own average (SD)"] = last.Yield_z
    t["Next-year outlook (kg/ha)"] = o1.Central
    c = D.cond.set_index("District")
    t["Crop status"] = c.Recent_status
    t["NDVI trend (30 days)"] = c.Trend
    t["Stage in 30 days"] = c.Stage_30d
    return t.rename_axis("Granary").reset_index()


def render(D):
    st.subheader("Overview")
    info_panel(
        "A one-page summary of every granary: the latest reported yield and how it moved, the current crop condition from "
        "satellite, and the outlook for next year. The other three tabs go deeper: Dynamic maps, Yield (past and outlook) "
        "and Advisory and intervention.",
        "Check how your granary is doing against the others, then open the Advisory tab for what to do about it.",
        "Spot which granaries are behind, falling or unusual and decide where to look first.",
        "yield is an annual granary-level figure (no planted area, so averages are not area-weighted); crop status comes from "
        "satellite NDVI and is a prompt for a field check, not a diagnosis.")
    sel = selected_granary()
    tbl = scoreboard(D)
    cond = D.cond.set_index("District")
    n_behind = int((cond.Recent_status == "behind").sum())
    n_fall = int((cond.Trend == "falling").sum())
    last = D.yh[D.yh.Year == D.LAST_YEAR]
    prev = D.yh[D.yh.Year == D.LAST_YEAR - 1].set_index("District").Yield

    k = st.columns(5)
    k[0].metric("Granaries", len(D.ALL))
    k[1].metric(f"Average yield {D.LAST_YEAR}", f"{last.Yield.mean():,.0f} kg/ha",
                f"{(last.Yield.mean() / prev.reindex(last.District).mean() - 1) * 100:+.1f}% vs {D.LAST_YEAR - 1}",
                delta_color="off", help="Simple average over granaries; not area-weighted.")
    k[2].metric("Highest yield", last.loc[last.Yield.idxmax(), "District"], f"{last.Yield.max():,.0f} kg/ha", delta_color="off")
    k[3].metric("Behind (crop status)", f"{n_behind} / {len(cond)}", help="Candidates for a field check.")
    k[4].metric("NDVI falling next 30 days", f"{n_fall} / {len(cond)}",
                help="Falling is normal after heading; read it with the stage.")

    # ---- staleness warning: computed from the data, so it disappears once the extraction is refreshed
    try:
        age = (pd.Timestamp.today().normalize() - pd.Timestamp(D.AS_OF)).days
    except (ValueError, TypeError):
        age = 0
    if age > 45:
        st.warning(f"The satellite data ends {D.AS_OF}, {age} days ago. Crop status, the 30-day NDVI forecast and the "
                   "advice are about that date, not about today. Re-run the extraction with a later --end-year to refresh.")

    mcol, lcol = st.columns([3, 1])
    m = (D.fp.merge(D.districts[["District", "Latitude", "Longitude"]], on="District")
         .merge(D.cond[["District", "Recent_status"]], on="District", how="left"))
    with mcol:
        st.markdown("**Recent crop status by granary** (click one to select it)")
        lo, hi, clicked, present = polygon_map(m, STATUS_SPEC, view_for(D), selected=sel, key="map_overview", height=440)
    with lcol:
        show_legend(STATUS_SPEC, lo, hi, present)
    apply_map_click(clicked, "overview")

    st.markdown("**Granary scoreboard**")
    st.dataframe(
        tbl, hide_index=True, width="stretch",
        column_config={
            f"Yield {D.LAST_YEAR} (kg/ha)": st.column_config.NumberColumn(format="%d"),
            "Change vs previous year (%)": st.column_config.NumberColumn(format="%+.1f"),
            "vs own average (SD)": st.column_config.NumberColumn(format="%+.2f",
                                                                  help="0 = a typical year for this granary"),
            "Next-year outlook (kg/ha)": st.column_config.NumberColumn(format="%d")})

    st.markdown("**Points to know before using the numbers**")
    shared = D.yh.dropna(subset=["Shares_series_with"])
    shared = shared[shared.Shares_series_with.astype(str).str.len() > 0].District.unique()
    jumps = D.yh[D.yh.Flag_large_jump]
    notes = []
    if len(shared):
        notes.append("Granaries reporting an identical yield series (not independent): " + ", ".join(shared))
    if len(jumps):
        worst = jumps.groupby("District").Year.apply(lambda s: ", ".join(map(str, s)))
        notes.append("Large year-to-year yield jumps (check how yield was counted): "
                     + "; ".join(f"{d} ({y})" for d, y in worst.items()))
    if D.WEAK:
        notes.append("Weak planting signal in the satellite series: " + ", ".join(D.WEAK))
    notes.append("On the maps each granary is the administrative district(s) it covers; the satellite values come only from "
                 "sampled paddy pixels inside them, not from the whole district.")
    for n in notes:
        st.markdown(f"- {n}")

    with st.expander("How reliable is this? (model track record and known limits)"):
        tab_reliability.render(D)
