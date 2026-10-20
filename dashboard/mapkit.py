"""Map helpers shared by the Maps tab and the Yield tab: colour ramp, legend, one pydeck polygon map per granary."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from . import adminmap

RAMP = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]     # viridis-like
GREY = [225, 225, 215, 255]                                   # a granary with no value for this layer: pale, but still clearly drawn
ALPHA = 235                                                   # strong fills so the colours read at a glance
SELECT_LINE = [220, 20, 60, 255]


def ramp(t):
    t = float(np.clip(t, 0, 1)) * (len(RAMP) - 1)
    i = min(int(t), len(RAMP) - 2)
    f = t - i
    return [int(RAMP[i][k] + f * (RAMP[i + 1][k] - RAMP[i][k])) for k in range(3)] + [ALPHA]


def swatch(rgb, label):
    return (f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;border:1px solid #666;"
            f"background:rgb({rgb[0]},{rgb[1]},{rgb[2]});vertical-align:middle;margin-right:6px'></span>{label}")


def view_for(D, selected=None):
    """(lat, lon, zoom) that fits all granaries, or just the selected one. Computed from the data, not hard-coded."""
    b = adminmap.bounds_of(selected)
    if b is not None:                                    # administrative shapes available
        if selected is None:
            return adminmap.peninsula_view()
        lon0, lat0, lon1, lat1 = b
    else:                                                # fall back to the sampled-pixel outlines
        fp = D.fp if selected is None else D.fp[D.fp.District == selected]
        pts = np.array([p for poly in fp.polygon for p in poly])
        (lon0, lat0), (lon1, lat1) = pts.min(0), pts.max(0)
    span = max(lon1 - lon0, lat1 - lat0, 0.35)
    zoom = float(np.clip(np.log2(800 * 360 / (512 * span)) - 0.8, 3.0, 11.0))
    return (lat0 + lat1) / 2, (lon0 + lon1) / 2, zoom


def polygon_map(frame, spec, view, *, selected=None, show_dots=False, show_names=True, height=520, key="map",
                vmin=None, vmax=None):
    """Each granary drawn as the administrative district(s) it covers (falls back to the outline of its sampled paddy
    pixels), over a state base map, coloured by frame[spec['col']].

    frame needs: District, polygon, Latitude, Longitude, the value column, optionally 'Note' (shown in the tooltip).
    vmin/vmax fix the colour scale so maps for different years stay comparable.
    Returns (lo, hi, clicked_granary, counts_of_categories).
    """
    col = spec["col"]
    d = frame[["District", "polygon", "Latitude", "Longitude", col]].copy()
    d["Note"] = frame["Note"].fillna("") if "Note" in frame else ""
    has = d[col].notna()
    lo = hi = None
    if spec["kind"] == "cont":
        if has.any():
            lo = float(d.loc[has, col].min()) if vmin is None else float(vmin)
            hi = float(d.loc[has, col].max()) if vmax is None else float(vmax)
            t = ((d[col] - lo) / (hi - lo)) if hi > lo else pd.Series(0.5, index=d.index)
        else:
            t = pd.Series(0.0, index=d.index)
        d["fill"] = [ramp(x) if h else GREY for x, h in zip(t.fillna(0), has)]
        d["label"] = [(spec["fmt"].format(v) + spec.get("unit", "") + (f" · {n}" if n else "")) if h else "no value"
                      for v, n, h in zip(d[col], d["Note"], has)]
    else:
        d["fill"] = [list(spec["colours"].get(v, (150, 150, 150))) + [ALPHA] if h else GREY for v, h in zip(d[col], has)]
        d["label"] = [str(v) if h else "no value" for v, h in zip(d[col], has)]
    present = d.loc[has, col].value_counts().to_dict()
    d = d.drop(columns=[col, "Note"])
    adm = adminmap.load()
    layers = []
    if adm is not None and not adm[0].empty:             # state base map + each granary as its district shapes
        parts, pts, base = adm
        have = set(parts.District)
        geo = parts.merge(d[d.District.isin(have)].drop(columns=["polygon", "Latitude", "Longitude"]), on="District")
        geo = geo.merge(pts[["District", "Latitude", "Longitude"]], on="District")
        rest = d[~d.District.isin(have)].assign(Admin="sampled paddy pixels (no district shape for this granary)")
        d = pd.concat([geo, rest], ignore_index=True)
        layers.append(pdk.Layer("PolygonLayer", base, id="states", get_polygon="polygon", get_fill_color=[233, 237, 243, 255],
                                get_line_color=[110, 120, 135, 255], line_width_min_pixels=1, pickable=False))
    else:
        d["Admin"] = "sampled paddy pixels"
    layers.append(pdk.Layer("PolygonLayer", d, id="granaries", get_polygon="polygon", get_fill_color="fill",
                            get_line_color=[30, 30, 30, 255], line_width_min_pixels=2, pickable=True, auto_highlight=True))
    if selected in set(d.District):                      # the shared selection gets a thick red outline
        layers.append(pdk.Layer("PolygonLayer", d[d.District == selected], id="selected", get_polygon="polygon",
                                filled=False, stroked=True, get_line_color=SELECT_LINE, line_width_min_pixels=5,
                                pickable=False))
    if show_dots:
        layers.append(pdk.Layer("ScatterplotLayer", d.drop_duplicates("District"), id="dots", get_position=["Longitude", "Latitude"],
                                get_fill_color="fill", get_line_color=[0, 0, 0, 255], stroked=True,
                                line_width_min_pixels=1, get_radius=3000, radius_min_pixels=7, radius_max_pixels=16,
                                pickable=True))
    if show_names:
        layers.append(pdk.Layer("TextLayer", d.drop_duplicates("District")[["District", "Latitude", "Longitude"]], id="names",
                                get_position=["Longitude", "Latitude"], get_text="District", get_size=14,
                                get_color=[15, 15, 15, 255], get_text_anchor="'middle'",
                                get_alignment_baseline="'top'", pickable=False, font_weight=700,
                                outline_width=3, outline_color=[255, 255, 255, 255], font_settings={"sdf": True}))
    deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=view[0], longitude=view[1], zoom=view[2]),
                    map_style="light_no_labels", tooltip={"text": "{District}\n{Admin}\n{label}"})
    clicked = None
    try:
        event = st.pydeck_chart(deck, height=height, on_select="rerun", selection_mode="single-object", key=key)
        objs = event.selection.objects.get("granaries", [])
        clicked = objs[0].get("District") if objs else None
    except TypeError:                                    # older Streamlit without map selection: still show the map
        st.pydeck_chart(deck, height=height)
    return lo, hi, clicked, present


def legend_figure(lo, hi, fmt):
    vals = np.linspace(hi, lo, 120)
    scale = [[i / (len(RAMP) - 1), f"rgb({c[0]},{c[1]},{c[2]})"] for i, c in enumerate(RAMP)]
    fig = go.Figure(go.Heatmap(z=vals.reshape(-1, 1), x=[0], y=vals, colorscale=scale, showscale=False, hoverinfo="skip"))
    ticks = [lo, (lo + hi) / 2, hi]
    fig.update_xaxes(visible=False)
    pad = (hi - lo) * 0.08 or 0.5                      # room so the lowest and highest labels are not clipped
    fig.update_yaxes(side="right", tickvals=ticks, ticktext=[fmt.format(t) for t in ticks], showgrid=False,
                     range=[lo - pad, hi + pad])
    fig.update_layout(height=260, margin=dict(l=0, r=60, t=5, b=5))
    return fig


def show_legend(spec, lo, hi, present, key=None):
    st.markdown("**How to read the colours**")
    if spec["kind"] == "cont":
        if lo is None:
            st.caption("No granary has a value for this layer.")
        else:
            st.plotly_chart(legend_figure(lo, hi, spec["fmt"]), width="stretch", key=key)
            st.markdown(swatch(RAMP[-1], f"**Yellow:** {spec['high']}") + "<br>" +
                        swatch(RAMP[0], f"**Purple:** {spec['low']}"), unsafe_allow_html=True)
    else:
        st.markdown("<br>".join(swatch(c, f"**{k}**" + (f" ({present.get(k, 0)})" if present.get(k, 0) else ""))
                                for k, c in spec["colours"].items()), unsafe_allow_html=True)
    st.markdown(swatch(GREY[:3], "**Pale:** no value") + "<br>"
                "<span style='display:inline-block;width:14px;height:14px;border:2px solid #282828;vertical-align:middle;"
                "margin-right:6px'></span>**Coloured area:** the administrative district(s) of one granary<br>"
                "<span style='display:inline-block;width:14px;height:14px;border:3px solid #dc1440;vertical-align:middle;"
                "margin-right:6px'></span>**Red outline:** the granary selected in the sidebar", unsafe_allow_html=True)
    st.caption(spec["meaning"])
