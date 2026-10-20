"""Component 3 · Yield and outlook, as two clearly separate sections.

  Yield   what was reported: a past year on the map, a ranking, and the granary's yield facts. Nothing else is mixed in.
  Outlook what to expect: the projection for the next years with its range, in its own section.
Each section has a button that shows the last records and the trend for the selected granary, only when the user asks for it.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .charts import BLUE, LIGHT_BLUE, yield_figure
from .mapkit import polygon_map, view_for
from .sidepanel import yield_panel
from .ui import apply_map_click, info_panel, selected_granary


def _spec(meaning):
    return dict(col="Yield", fmt="{:,.0f}", unit=" kg/ha", kind="cont", low="lower yield", high="higher yield", meaning=meaning)


PAST_SPEC = _spec("Reported yield. The colour scale is the same for reported yield and the outlook, so the two sections can be compared.")
OUT_SPEC = _spec("Outlook yield: a projection, not a measurement. Same colour scale as the reported yield.")


def _base(D):
    return D.fp.merge(D.districts[["District", "Latitude", "Longitude"]], on="District")


def past_frame(D, year):
    v = D.yh[D.yh.Year == year][["District", "Yield"]].assign(Kind="observed", Low=np.nan, High=np.nan)
    m = _base(D).merge(v, on="District", how="left")
    m["Note"] = np.where(m.Yield.notna(), "reported", "")
    return m


def outlook_frame(D, year):
    v = (D.outlook[D.outlook.Forecast_year == year][["District", "Central", "Low", "High"]]
         .rename(columns={"Central": "Yield"}).assign(Kind="outlook"))
    m = _base(D).merge(v, on="District", how="left")
    m["Note"] = [("" if pd.isna(lo) else f"outlook, range {lo:,.0f} to {hi:,.0f}") for lo, hi in zip(m.Low, m.High)]
    return m


def ranking_figure(frame, title):
    srt = frame.dropna(subset=["Yield"]).sort_values("Yield")
    colours = [BLUE if k == "observed" else LIGHT_BLUE for k in srt.Kind]
    fig = go.Figure(go.Bar(
        x=srt.Yield, y=srt.District, orientation="h", marker_color=colours,
        error_x=dict(type="data", symmetric=False, array=(srt.High - srt.Yield).fillna(0),
                     arrayminus=(srt.Yield - srt.Low).fillna(0), color="#999"),
        hovertemplate="%{y}: %{x:,.0f} kg/ha<extra></extra>"))
    fig.update_layout(height=max(260, 30 * len(srt) + 90), xaxis_title=title, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def history_figure(D, g, compare=(), mark=None):
    """reported yield only: the granary's records year by year, its own average, jumps, and optionally other granaries"""
    d = D.yh[D.yh.District == g].sort_values("Year")
    fig = go.Figure()
    for c in compare:
        dc = D.yh[D.yh.District == c].sort_values("Year")
        fig.add_trace(go.Scatter(x=dc.Year, y=dc.Yield, mode="lines", name=c, line=dict(width=1.5), opacity=0.6))
    fig.add_trace(go.Scatter(x=d.Year, y=d.Yield, mode="lines+markers", name=f"{g}: reported", line=dict(color=BLUE, width=3)))
    jump = d[d.Flag_large_jump]
    if len(jump):
        fig.add_trace(go.Scatter(x=jump.Year, y=jump.Yield, mode="markers", name="Change of 20% or more",
                                 marker=dict(color="red", size=14, symbol="circle-open", line=dict(width=2))))
    if g in D.MEAN_YIELD.index:
        fig.add_hline(y=float(D.MEAN_YIELD[g]), line_dash="dot", line_color="#999", annotation_text="granary average",
                      annotation_position="bottom left")
    if mark is not None:
        fig.add_vline(x=mark, line_dash="dot", line_color="#555")
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=360, yaxis_title="Yield (kg/ha)", xaxis_title="Year", margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def _toggle(key, show_label, hide_label):
    """a button that opens or closes a block; closed until the user asks"""
    on = st.session_state.get(key, False)
    if st.button(hide_label if on else show_label, key=key + "_btn"):
        st.session_state[key] = not on
        st.rerun()
    return st.session_state.get(key, False)


def _records(D, g, with_outlook, key):
    """the last records of one granary as a trend chart and a table, with the outlook added when asked"""
    compare = st.multiselect("Compare with other granaries (reported yield)", [x for x in D.ALL if x != g], key=key + "_cmp")
    fig = yield_figure(D, g, height=380, compare=compare) if with_outlook else history_figure(D, g, compare)
    st.plotly_chart(fig, width="stretch", key=key + "_fig")
    d = D.yh[D.yh.District == g].sort_values("Year", ascending=False)
    t = pd.DataFrame({"Year": d.Year, "Yield (kg/ha)": d.Yield.round(0), "Change on previous year (%)": d.Change_pct.round(1),
                      "Against own average (SD)": d.Yield_z.round(2),
                      "Note": np.where(d.Flag_large_jump, "change of 20% or more: check how yield was counted", "")})
    st.dataframe(t, hide_index=True, width="stretch")
    shared = d.Shares_series_with.iloc[0] if len(d) else ""
    if isinstance(shared, str) and shared:
        st.warning(f"{g} reports the same yield series as: {shared}. They are not independent measurements.")


def _controls(D, key, options, label, fmt=str, default=None):
    c1, c2, c3 = st.columns([5, 1, 2])
    year = c1.select_slider(label, options=options, value=default if default is not None else options[0], format_func=fmt, key=f"{key}_year")
    names = c2.checkbox("Names", value=True, key=f"{key}_names")
    zoom = c3.radio("Zoom to", ["All granaries", "Selected granary"], horizontal=True, key=f"{key}_zoom")
    return year, names, zoom


# ------------------------------------------------------------------------------------------------------------- yield
def _yield_section(D, sel, vmin, vmax):
    st.markdown("### Yield: what was reported")
    st.caption("Annual yield for each granary. Pick a year; click a granary on the map to see its figures beside it.")
    years = sorted(int(y) for y in D.yh.Year.unique())
    year, names, zoom = _controls(D, "yld", years, "Year", default=years[-1])
    frame = past_frame(D, year)
    v = frame.dropna(subset=["Yield"]).set_index("District").Yield
    prev = D.yh[D.yh.Year == year - 1].set_index("District").Yield
    common = v.index.intersection(prev.index)
    k = st.columns(4)
    k[0].metric(f"Average yield {year}", f"{v.mean():,.0f} kg/ha" if len(v) else "n/a",
                help="Simple average over granaries; planted area is not available, so it is not area-weighted.")
    k[1].metric(f"Change on {year - 1}", f"{(v[common].mean() / prev[common].mean() - 1) * 100:+.1f}%" if len(common) else "n/a")
    if len(v):
        k[2].metric("Highest", v.idxmax(), f"{v.max():,.0f} kg/ha", delta_color="off")
        k[3].metric("Lowest", v.idxmin(), f"{v.min():,.0f} kg/ha", delta_color="off")
    mcol, lcol = st.columns([3, 2])
    with mcol:
        view = view_for(D, sel if zoom == "Selected granary" else None)
        lo, hi, clicked, present = polygon_map(frame, PAST_SPEC, view, selected=sel, show_names=names, key="map_yield_past",
                                               vmin=vmin, vmax=vmax, height=520)
    with lcol:
        yield_panel(D, sel, PAST_SPEC, lo, hi, present, 520, "past", year)
    apply_map_click(clicked, "yield_past")
    st.plotly_chart(ranking_figure(frame, f"Reported yield {year} (kg/ha)"), width="stretch", key="yld_rank")
    st.markdown(f"**Last records for {sel}**")
    if _toggle("yld_trend", f"Show the last records and trend for {sel}", "Hide the last records"):
        _records(D, sel, False, "yld_rec")


# ----------------------------------------------------------------------------------------------------------- outlook
def _outlook_section(D, sel, vmin, vmax):
    st.markdown("### Outlook: what to expect")
    st.caption("A projection of yield for the next years with a range. It is a planning range, not a guarantee.")
    years = sorted(int(y) for y in D.outlook.Forecast_year.unique())
    fmt = lambda y: f"{y}  ({y - D.LAST_YEAR} year{'s' if y - D.LAST_YEAR > 1 else ''} ahead)"  # noqa: E731
    year, names, zoom = _controls(D, "out", years, "Outlook year", fmt, years[0])
    frame = outlook_frame(D, year)
    v = frame.dropna(subset=["Yield"]).set_index("District").Yield
    o = D.outlook[D.outlook.Forecast_year == year]
    ref = o.set_index("District").Last_yield
    common = v.index.intersection(ref.index)
    k = st.columns(4)
    k[0].metric(f"Average outlook {year}", f"{v.mean():,.0f} kg/ha" if len(v) else "n/a", help="Simple average over granaries; not area-weighted.")
    k[1].metric("Change on latest reported", f"{(v[common].mean() / ref[common].mean() - 1) * 100:+.1f}%" if len(common) else "n/a")
    k[2].metric("Typical error", f"±{o.Band_pct.iloc[0]:.0f}%", help="80th percentile of the errors this method made in back-tests at this distance.")
    k[3].metric("Years ahead", int(o.Horizon_years.iloc[0]))
    st.caption(f"Method: {o.Method.iloc[0]}")
    mcol, lcol = st.columns([3, 2])
    with mcol:
        view = view_for(D, sel if zoom == "Selected granary" else None)
        lo, hi, clicked, present = polygon_map(frame, OUT_SPEC, view, selected=sel, show_names=names, key="map_yield_outlook",
                                               vmin=vmin, vmax=vmax, height=520)
    with lcol:
        yield_panel(D, sel, OUT_SPEC, lo, hi, present, 520, "outlook", year)
    apply_map_click(clicked, "yield_outlook")
    st.plotly_chart(ranking_figure(frame, f"Outlook {year} (kg/ha); grey lines show the range"), width="stretch", key="out_rank")

    st.markdown(f"**{sel}: outlook for every year ahead**")
    og = D.outlook[D.outlook.District == sel].sort_values("Forecast_year")
    if len(og):
        st.dataframe(pd.DataFrame({
            "Year": og.Forecast_year, "Years ahead": og.Horizon_years, "Outlook (kg/ha)": og.Central.round(0),
            "Range from": og.Low.round(0), "Range to": og.High.round(0), "Typical error (%)": og.Band_pct.round(0),
            "Change on latest reported (%)": ((og.Central / og.Last_yield - 1) * 100).round(1)}),
            hide_index=True, width="stretch")
    if _toggle("out_trend", f"Show the last records with the outlook for {sel}", "Hide the last records"):
        _records(D, sel, True, "out_rec")
    st.info("These are planning ranges from a method that, in back-tests, was about as accurate as 'same as last year': climate, management, "
            "policy and cropping conditions can change the actual yield.")
    with st.expander("How the outlook is made"):
        how = D.outlook.groupby("Horizon_years").agg(Method=("Method", "first"), Typical_error_pct=("Band_pct", "first")).reset_index()
        st.dataframe(how, hide_index=True, width="stretch")
        st.markdown("The carry-over rate (how much one year's yield carries into the next) is learned from the other granaries and applied to "
                    "each granary. The Model check tab repeats the test with and without transfer learning. Source file: "
                    "`data/yield_outlook.csv`.")
    with st.expander("Table: every granary, reported and outlook (kg/ha)"):
        obs = D.yh.pivot(index="District", columns="Year", values="Yield")
        obs.columns = [str(c) for c in obs.columns]
        fc = D.outlook.pivot(index="District", columns="Forecast_year", values="Central")
        fc.columns = [f"{c} (outlook)" for c in fc.columns]
        st.dataframe(obs.join(fc).round(0).reset_index(), hide_index=True, width="stretch")


def render(D):
    st.subheader("Yield and outlook")
    info_panel(
        "Two separate sections. Yield shows what each granary reported, year by year. Outlook shows what to expect in the next 1 to 4 years, "
        "with a range. Each has its own map and ranking, and a button that shows the last records and the trend of the selected granary.",
        "",
        "Use Yield to see which granaries are high or low and how stable they are; use Outlook as a planning range for production, storage "
        "and inputs. Check granaries with sudden changes at the source.",
        "the outlook is only as accurate as 'same as last year' in back-tests (about 8% error at 1 year, 11% at 2, 14% at 3), and the "
        "reported yield has only 7 yearly values per granary.")
    sel = selected_granary()
    vmin = min(D.yh.Yield.min(), D.outlook.Low.min())
    vmax = max(D.yh.Yield.max(), D.outlook.High.max())
    t_yield, t_out = st.tabs(["📊 Yield: what was reported", "🔮 Outlook: what to expect"])
    with t_yield:
        _yield_section(D, sel, vmin, vmax)
    with t_out:
        _outlook_section(D, sel, vmin, vmax)
