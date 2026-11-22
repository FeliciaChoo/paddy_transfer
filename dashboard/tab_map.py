import html as _h
import numpy as np
import pandas as pd
import streamlit as st
import folium
from branca.element import MacroElement, Template
from streamlit_folium import st_folium

# 6-Tier Operational Screening Condition Palette (Matches PaddyTwinAI Sample)
CONDITION_META = {
    "Inspect declining vegetation": {
        "color": "#DC2626",
        "expected": "NDVI dropped below forecast range in recent decades",
        "action": "Inspect field bunds, pest incidence, and water stress before next fertilizing round.",
    },
    "Limited satellite visibility": {
        "color": "#8B95A5",
        "expected": "Insufficient / cloud-obscured optical data (< 45% valid coverage)",
        "action": "An assigned drain/bund clearance list and a revised weather-sensitive work roster.",
    },
    "Confirm establishment": {
        "color": "#EAB308",
        "expected": "Early vegetative / transplanting window (0–30 HST)",
        "action": "Verify uniform crop establishment, seedling survival, and early water level (3–5 cm).",
    },
    "Review harvest progress": {
        "color": "#7E22CE",
        "expected": "Late ripening / maturity stage (> 85 HST)",
        "action": "Drain fields 10–14 days pre-harvest and schedule combine harvester logistics.",
    },
    "Maintain field surveillance": {
        "color": "#0E7490",
        "expected": "Reproductive / heading transition (40–70 HST)",
        "action": "Maintain 5–10 cm standing water depth and weekly IPM pest scouting.",
    },
    "Routine monitoring": {
        "color": "#15803D",
        "expected": "Canopy greenness tracking within normal seasonal band",
        "action": "Continue routine irrigation schedule and standard NPK/Urea split application.",
    },
}

YIELD_COLOURS = ["#DC2626", "#F59E0B", "#FEF08A", "#86EFAC", "#15803D"]
FIXED_YIELD_EDGES = [2.46, 3.04, 3.63, 4.22]
NO_DATA_COLOUR = "#8B95A5"

GRANARY_META = {
    "MADA": {"coords": [6.12, 100.37], "state": "Kedah & Perlis", "s1": "Mar – Jul", "s2": "Sep – Jan"},
    "IADA Pulau Pinang": {"coords": [5.42, 100.43], "state": "Pulau Pinang", "s1": "Apr – Aug", "s2": "Oct – Feb"},
    "IADA Kerian": {"coords": [5.01, 100.48], "state": "Perak", "s1": "Mar – Jul", "s2": "Sep – Jan"},
    "IADA Seberang Perak": {"coords": [4.10, 100.95], "state": "Perak", "s1": "Mar – Jul", "s2": "Sep – Jan"},
    "IADA Barat Laut Selangor": {"coords": [3.55, 101.10], "state": "Selangor", "s1": "Jan – Apr (4 Blocks)", "s2": "Jul – Oct (4 Blocks)"},
    "KADA": {"coords": [6.05, 102.28], "state": "Kelantan", "s1": "May – Sep", "s2": "Nov – Mar"},
    "IADA Kemasin Semerak": {"coords": [5.95, 102.42], "state": "Kelantan", "s1": "May – Sep", "s2": "Nov – Mar"},
    "KETARA": {"coords": [5.68, 102.62], "state": "Terengganu", "s1": "May – Sep", "s2": "Nov – Mar"},
    "IADA KETARA": {"coords": [5.68, 102.62], "state": "Terengganu", "s1": "May – Sep", "s2": "Nov – Mar"},
    "IADA Pekan": {"coords": [3.50, 103.38], "state": "Pahang", "s1": "Apr – Aug", "s2": "Oct – Feb"},
    "IADA Rompin": {"coords": [2.80, 103.48], "state": "Pahang", "s1": "Apr – Aug", "s2": "Oct – Feb"},
}


def _derive_condition(row) -> str:
    """Derive operational screening condition from Optical_Valid_Fraction, Recent_status, or Yield_z."""
    ovf = row.get("Optical_Valid_Fraction", np.nan)
    if pd.notna(ovf) and ovf < 0.45:
        return "Limited satellite visibility"

    status = str(row.get("Recent_status", "")).strip().lower()
    if status == "behind":
        return "Inspect declining vegetation"
    if status == "ahead":
        return "Maintain field surveillance"

    yz = row.get("Yield_z", np.nan)
    if pd.notna(yz):
        if yz < -0.75:
            return "Inspect declining vegetation"
        if yz > 0.75:
            return "Maintain field surveillance"
    return "Routine monitoring"


def _yield_colour(val: float) -> str:
    if pd.isna(val):
        return NO_DATA_COLOUR
    for edge, col in zip(FIXED_YIELD_EDGES, YIELD_COLOURS):
        if val < edge:
            return col
    return YIELD_COLOURS[-1]


def _ndvi_colour(val: float) -> str:
    if pd.isna(val):
        return NO_DATA_COLOUR
    if val < 0.35:
        return "#EAB308"
    if val < 0.50:
        return "#86EFAC"
    if val < 0.65:
        return "#15803D"
    return "#0E7490"


def _build_tooltip(r, year: int) -> str:
    t_ha_txt = f"{r['Yield_t_ha']:.2f} t/ha ({r['Yield_kg_ha']:,.0f} kg/ha)" if pd.notna(r["Yield_t_ha"]) else "No reference"
    ovf_txt = f"{100 * r['Optical_Valid_Fraction']:.1f}%" if pd.notna(r["Optical_Valid_Fraction"]) else "N/A"
    ndvi_txt = f"{r['Last_NDVI']:.2f}" if pd.notna(r.get("Last_NDVI", np.nan)) else "N/A"
    return (
        f'<div style="font-family:Inter,sans-serif;font-size:13px;color:#0B2117;min-width:225px;">'
        f'<div style="font-weight:700;font-size:14px;color:#15803D;margin-bottom:4px;">'
        f'{_h.escape(str(r["District"]))} ({_h.escape(str(r["State"]))})</div>'
        f'<div><b>Screening Condition:</b> {_h.escape(str(r["Condition"]))}</div>'
        f'<div><b>Annual Yield ({year}):</b> {t_ha_txt}</div>'
        f'<div><b>Last NDVI / Optical Valid:</b> {ndvi_txt} / {ovf_txt}</div>'
        f'<div><b>Planting Windows:</b> S1 {_h.escape(str(r["Season_1_Window"]))} | S2 {_h.escape(str(r["Season_2_Window"]))}</div>'
        f'<div style="margin-top:4px;font-size:11px;color:#475569;">Click field for register details</div>'
        f"</div>"
    )


def render_leaflet_map(D=None, role="kpkm", title="Paddy Field Conditions & Spatial Yield Map"):
    st.markdown(f"## {title}")
    st.caption("Review satellite conditions, annual yield benchmarks, and planting-calendar estimates before planning field visits.")

    # Load dataset from D.DATA_DIR or fallback path
    data_dir = getattr(D, "DATA_DIR", None)
    stage_path = (data_dir / "transfer_stage_dataset.csv") if data_dir else pd.io.common.stringify_path("data/transfer_stage_dataset.csv")
    yield_path = (data_dir / "transfer_yield_dataset.csv") if data_dir else pd.io.common.stringify_path("data/transfer_yield_dataset.csv")

    try:
        df = pd.read_csv(stage_path)
    except Exception:
        df = pd.read_csv(yield_path)

    if "Optical_Valid_Fraction" not in df.columns:
        df["Optical_Valid_Fraction"] = df["Optical_observed_pct"] / 100.0 if "Optical_observed_pct" in df.columns else np.nan

    years = sorted(int(y) for y in df["Year"].unique())
    selected_year = st.session_state.get("map_year", years[-1])

    df_yr = df[df["Year"] == selected_year].copy()
    df_yr["Yield_t_ha"] = df_yr["Yield_kg_ha"] / 1000.0

    # Merge coordinates & metadata
    coords_map = {}
    if D is not None and hasattr(D, "districts"):
        for _, r in D.districts.iterrows():
            coords_map[str(r["District"])] = (float(r["Latitude"]), float(r["Longitude"]))

    if D is not None and hasattr(D, "cond") and D.cond is not None and len(D.cond):
        df_yr = df_yr.merge(D.cond[["District", "Last_NDVI", "Recent_status"]], on="District", how="left")
    else:
        df_yr["Last_NDVI"] = np.nan
        df_yr["Recent_status"] = ""

    rows = []
    for _, r in df_yr.iterrows():
        dist = str(r["District"])
        meta = GRANARY_META.get(dist, {"coords": [4.2, 101.5], "state": "Malaysia", "s1": "Mar – Jul", "s2": "Sep – Jan"})
        lat, lon = coords_map.get(dist, tuple(meta["coords"]))
        cond = _derive_condition(r)
        cmeta = CONDITION_META[cond]
        rows.append({
            "District": dist,
            "State": meta["state"],
            "Latitude": lat,
            "Longitude": lon,
            "Year": selected_year,
            "Yield_kg_ha": r["Yield_kg_ha"],
            "Yield_t_ha": r["Yield_t_ha"],
            "Optical_Valid_Fraction": r["Optical_Valid_Fraction"],
            "Last_NDVI": r["Last_NDVI"],
            "Condition": cond,
            "Expected_result": cmeta["expected"],
            "Action_basis": cmeta["action"],
            "Season_1_Window": meta["s1"],
            "Season_2_Window": meta["s2"],
        })
    reg_df = pd.DataFrame(rows)

    # Top KPI Strip (Matches PaddyTwinAI Sample Page 1)
    usable_count = int((reg_df["Optical_Valid_Fraction"].fillna(0) >= 0.45).sum())
    priority_count = int(reg_df["Condition"].isin(["Inspect declining vegetation", "Limited satellite visibility"]).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Mapped paddy field groups", f"{len(reg_df)}")
    k2.metric("With usable satellite samples", f"{usable_count}")
    k3.metric("Priority work / visits", f"{priority_count}")

    # Filter Row (State, Satellite Screening Status, Reference Year)
    f1, f2, f3 = st.columns(3)
    state_filter = f1.multiselect("State", sorted(reg_df["State"].unique()), default=[], placeholder="Choose options")
    cond_filter = f2.multiselect("Satellite screening status", list(CONDITION_META.keys()), default=[], placeholder="Choose options")
    selected_year = f3.selectbox("Reference Year", sorted(years, reverse=True), index=0, key="map_year")

    filtered = reg_df.copy()
    if state_filter:
        filtered = filtered[filtered["State"].isin(state_filter)]
    if cond_filter:
        filtered = filtered[filtered["Condition"].isin(cond_filter)]

    # Colour Mode & Action Bar (Matches PaddyTwinAI Sample Page 1 & 2)
    b1, b2, b3 = st.columns([1.4, 2.2, 2.4])
    with b1:
        colour_mode = st.radio("Colour fields by", ["Condition", "NDVI", "Annual Yield"], horizontal=True, key="map_colour_mode")
    with b2:
        if st.button("Show all mapped fields", key="map_show_all_btn"):
            st.session_state["map_fit_all"] = True
    with b3:
        st.caption("Hover for annual yield & condition. Click a field for details.")

    # Build Full-Malaysia OpenStreetMap Leaflet Map (100% Free, Zero API Key)
    m = folium.Map(location=[4.2, 108.5], zoom_start=6, tiles="OpenStreetMap", control_scale=False, zoom_control=True)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Clean Terrain / Topo",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Satellite Imagery",
        overlay=False,
        control=True,
    ).add_to(m)

    # Optional polygon overlay if D.fp GeoDataFrame is available
    if D is not None and hasattr(D, "fp") and D.fp is not None and len(D.fp):
        try:
            folium.GeoJson(
                D.fp[D.fp["District"].isin(filtered["District"])],
                name="Mapped Scheme Polygons",
                style_function=lambda _: {"fillColor": "#0F172A", "color": "#0F172A", "weight": 1.5, "fillOpacity": 0.45},
            ).add_to(m)
        except Exception:
            pass

    sel = st.session_state.get("granary")
    for _, r in filtered.iterrows():
        if colour_mode == "Condition":
            fill = CONDITION_META[r["Condition"]]["color"]
        elif colour_mode == "NDVI":
            fill = _ndvi_colour(r["Last_NDVI"])
        else:
            fill = _yield_colour(r["Yield_t_ha"])

        loc = [float(r["Latitude"]), float(r["Longitude"])]
        if r["District"] == sel:
            folium.CircleMarker(location=loc, radius=14, color="#15803D", weight=3, fill=False, interactive=False).add_to(m)

        folium.CircleMarker(
            location=loc,
            radius=8,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color=fill,
            fill_opacity=0.95,
            tooltip=folium.Tooltip(_build_tooltip(r, selected_year), sticky=True),
        ).add_to(m)

    if st.session_state.get("map_fit_all") and len(filtered):
        m.fit_bounds([[float(a), float(b)] for a, b in zip(filtered["Latitude"], filtered["Longitude"])], padding=(40, 40))

    # Bottom-Left Floating Legend matching selected colour mode
    if colour_mode == "Condition":
        leg_title = "Operational screening condition"
        leg_items = [(meta["color"], label) for label, meta in CONDITION_META.items()]
    elif colour_mode == "NDVI":
        leg_title = "Latest NDVI greenness"
        leg_items = [
            ("#EAB308", "< 0.35 (Early / sparse canopy)"),
            ("#86EFAC", "0.35 – 0.50 (Vegetative)"),
            ("#15803D", "0.50 – 0.65 (Dense canopy)"),
            ("#0E7490", "> 0.65 (Peak greenness)"),
            (NO_DATA_COLOUR, "Limited satellite visibility"),
        ]
    else:
        leg_title = f"Annual state reference - {selected_year} (t/ha)"
        b = [1.87] + FIXED_YIELD_EDGES + [4.81]
        leg_items = [(YIELD_COLOURS[i], f"{b[i]:.2f}–{b[i+1]:.2f} t/ha") for i in range(5)] + [(NO_DATA_COLOUR, "No state reference")]

    leg_rows = "".join(
        f'<div style="margin-bottom:3px;"><span style="background:{c};width:12px;height:12px;display:inline-block;margin-right:8px;border-radius:2px;vertical-align:middle;"></span>{lbl}</div>'
        for c, lbl in leg_items
    )
    legend = MacroElement()
    legend._template = Template(
        "{% macro html(this, kwargs) %}"
        '<div style="position:fixed;bottom:22px;left:22px;z-index:9999;background:rgba(255,255,255,0.97);'
        'border:1px solid #DCE3D3;border-radius:10px;padding:12px 16px;font-family:Inter,sans-serif;font-size:12px;'
        'color:#0B2117;box-shadow:0 4px 14px rgba(0,0,0,0.12);">'
        f'<div style="font-weight:700;margin-bottom:6px;">{leg_title}</div>{leg_rows}</div>'
        "{% endmacro %}"
    )
    m.get_root().add_child(legend)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    map_state = st_folium(
        m,
        key="leaflet_paddytwin_map",
        use_container_width=True,
        height=520,
        returned_objects=["last_object_clicked_tooltip"],
    )

    # Sync clicked marker to Selected Granary
    if map_state and map_state.get("last_object_clicked_tooltip"):
        tip_txt = str(map_state["last_object_clicked_tooltip"])
        for dist_name in sorted(filtered["District"].unique(), key=len, reverse=True):
            if dist_name in tip_txt and st.session_state.get("granary") != dist_name:
                st.session_state["granary"] = dist_name
                st.rerun()

    # Register Table Directly Below Map (Matches PaddyTwinAI Sample Page 2)
    st.caption("Click a field to select its group in the register and details below. State and condition filters apply to both map and register.")
    display_cols = filtered[["District", "State", "Condition", "Expected_result", "Optical_Valid_Fraction", "Action_basis"]].copy()
    display_cols.columns = ["Paddy Field", "State", "Condition", "Expected result", "Satellite evidence", "Action basis"]
    display_cols["Satellite evidence"] = display_cols["Satellite evidence"].apply(
        lambda v: f"{v*100:.0f}% clear-sky valid coverage" if pd.notna(v) else "Insufficient / older data"
    )
    st.dataframe(display_cols, hide_index=True, use_container_width=True)

    # Selected Field Action Card (Matches PaddyTwinAI Sample Page 3)
    active_field = st.session_state.get("granary")
    if active_field and active_field in filtered["District"].values:
        frow = filtered[filtered["District"] == active_field].iloc[0]
        st.markdown(f"### Review a paddy field · {frow['District']} ({frow['State']})")
        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:1px solid #DCE3D3;border-radius:14px;padding:18px;">
                <div style="font-weight:700;color:#0B2117;margin-bottom:6px;">
                    Recommended officer follow-up ({frow['Condition']}):
                </div>
                <div style="color:#0B2117;margin-bottom:8px;">{frow['Action_basis']}</div>
                <div style="font-size:12.5px;color:#43544B;">
                    <b>Typical regional window:</b> Season 1 ({frow['Season_1_Window']}) &nbsp;|&nbsp; Season 2 ({frow['Season_2_Window']})<br>
                    <b>Knowledge reference:</b> KPKM Staggered Paddy Planting Schedule &amp; DOA Manual Tanaman Padi
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
@st.cache_data(show_spinner=False)
def _map_table(path_stage, path_yield, mtime):
    """Return granary-year yield and Optical_Valid_Fraction for Overview and Map tabs."""
    import os
    if os.path.exists(path_stage):
        d = pd.read_csv(path_stage, usecols=["District", "Year", "Yield_kg_ha", "Optical_Valid_Fraction"])
    else:
        d = pd.read_csv(path_yield)
        d["Optical_Valid_Fraction"] = (
            d["Optical_observed_pct"] / 100.0 if "Optical_observed_pct" in d else np.nan
        )
        d = d[["District", "Year", "Yield_kg_ha", "Optical_Valid_Fraction"]]
    return d