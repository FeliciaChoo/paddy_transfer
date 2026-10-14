"""Plotly figures used by more than one tab."""
import pandas as pd
import plotly.graph_objects as go

BLUE, LIGHT_BLUE, GREEN, RED = "#1f77b4", "#7ab8f5", "#2ca02c", "#d62728"


def yield_figure(D, d, height=380, compare=(), marker_year=None):
    """Past yield (solid) and future outlook with its band (dashed) for one granary; optional faint lines for others."""
    g = D.yh[D.yh.District == d].sort_values("Year")
    o = D.outlook[D.outlook.District == d].sort_values("Forecast_year")
    fig = go.Figure()
    for c in compare:
        gc = D.yh[D.yh.District == c].sort_values("Year")
        fig.add_trace(go.Scatter(x=gc.Year, y=gc.Yield, mode="lines", name=c, line=dict(width=1.5), opacity=0.6))
    if len(o):
        fy = list(o.Forecast_year)
        fig.add_trace(go.Scatter(x=fy + fy[::-1], y=list(o.High) + list(o.Low)[::-1], fill="toself",
                                 fillcolor="rgba(255,120,120,0.25)", line=dict(width=0), name="Outlook band",
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g.Year, y=g.Yield, mode="lines+markers", name=f"{d}: reported", line=dict(color=BLUE)))
    if len(o):
        fig.add_trace(go.Scatter(x=[o.Last_year.iloc[0]] + list(o.Forecast_year), y=[o.Last_yield.iloc[0]] + list(o.Central),
                                 mode="lines+markers", name=f"{d}: outlook", line=dict(color=LIGHT_BLUE, dash="dash"),
                                 marker=dict(symbol="diamond")))
    jump = g[g.Flag_large_jump]
    if len(jump):
        fig.add_trace(go.Scatter(x=jump.Year, y=jump.Yield, mode="markers", name="Jump of 20% or more",
                                 marker=dict(color="red", size=14, symbol="circle-open", line=dict(width=2))))
    if d in D.MEAN_YIELD.index:
        fig.add_hline(y=float(D.MEAN_YIELD[d]), line_dash="dot", line_color="#999", annotation_text="granary average",
                      annotation_position="bottom left")
    if marker_year is not None:
        fig.add_vline(x=marker_year, line_dash="dot", line_color="#555")
    fig.update_layout(height=height, yaxis_title="Yield (kg/ha)", xaxis_title="Year", margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def ndvi_figure(D, d, height=260):
    """Observed NDVI history plus the 10- and 30-day forecast (with its 80% range). None if there is nothing to draw."""
    fig = go.Figure()
    drawn = False
    if D.series is not None and d in set(D.series.District):
        sd = D.series[D.series.District == d]
        fig.add_trace(go.Scatter(x=pd.to_datetime(sd.Date), y=sd.NDVI, mode="lines", name="Observed NDVI",
                                 line=dict(color=GREEN)))
        drawn = True
    s = D.cond[D.cond.District == d]
    if len(s):
        s = s.iloc[0]
        t0 = pd.to_datetime(s.As_of)
        t10, t30 = t0 + pd.Timedelta(days=10), t0 + pd.Timedelta(days=30)
        fig.add_trace(go.Scatter(x=[t0, t10, t30], y=[s.Last_NDVI, s.Forecast_10d, s.Forecast_30d], mode="lines+markers",
                                 name="Forecast", line=dict(color=RED, dash="dash")))
        fig.add_trace(go.Scatter(x=[t30], y=[s.Forecast_30d], mode="markers", marker=dict(opacity=0), showlegend=False,
                                 name="80% range (30 days)",
                                 error_y=dict(type="data", symmetric=False, array=[s.Upper_30d - s.Forecast_30d],
                                              arrayminus=[s.Forecast_30d - s.Lower_30d], color="#d62728")))
        drawn = True
    if not drawn:
        return None
    fig.update_layout(height=height, yaxis_title="NDVI", margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.3))
    return fig
