"""Component 6 · Model check: does transfer learning help? First the result WITHOUT it, then a button that applies it, then the improvement.

What is trained and what is not
  NDVI forecast (10 and 30 days)  trained model (ridge / gradient boosting on the 10-day satellite series of all granaries together)
  Yield outlook (1 to 3 years)    transfer learning of how yield behaves from year to year: the carry-over rate is learned on the OTHER granaries
  Advisory and intervention       NOT a trained model: editable rules (with sources), optional AI wording of the result

The yield part is run LIVE when the button is pressed (it takes about half a second): the back-test is repeated with and without transfer
learning on the reported yields. TrAdaBoost is too slow to repeat here, so its results are the stored ones from pipeline/transfer_benchmark.py
and pipeline/yield_transfer_benchmark.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from yield_dynamics_transfer import boot, ewma_pred, fit_alpha, fit_phi, run_backtest  # noqa: E402

from .ui import info_panel, selected_granary  # noqa: E402

GREY, BLUE, GREEN = "#9aa3ad", "#2b6cb0", "#2f855a"
HLAB = {1: "1 year ahead", 2: "2 years ahead", 3: "3 years ahead"}


@st.cache_data(show_spinner=False)
def _backtest(yh, transfer):
    return run_backtest(yh, horizons=(1, 2, 3), transfer=transfer)


def _table(bt, cols, names):
    t = bt.pivot_table(index="horizon", columns="method", values="ape", aggfunc="mean")[cols].round(1)
    t.insert(0, "Cases tested", bt.drop_duplicates(["horizon", "District", "test_year"]).groupby("horizon").size())
    t.index = [HLAB[h] for h in t.index]
    return t.rename(columns=names).rename_axis("Outlook")


def _verdict(lo, hi):
    return "Improved" if hi < 0 else ("Worse" if lo > 0 else "No clear difference")


def _own_outlook(series, bt_before):
    y = series.Yield.values
    out = {}
    for h in (1, 2, 3):
        a = fit_alpha([y], h) if len(y) >= h + 3 else 0.5
        band = np.percentile(bt_before[(bt_before.method == "ewma_own") & (bt_before.horizon == h)].ape, 80)
        out[h] = (ewma_pred(y, a), band)
    return out


def _tl_outlook(D, g, bt_after):
    S = {k: d.sort_values("Year").Yield.values for k, d in D.yh.groupby("District")}
    y, m = S[g], S[g].mean()
    out = {}
    for h in (1, 2, 3):
        phi = fit_phi([v for k, v in S.items() if k != g], h)
        band = np.percentile(bt_after[(bt_after.method == "ar1_transfer") & (bt_after.horizon == h)].ape, 80)
        out[h] = (m + phi * (y[-1] - m), band, phi)
    return out


def render(D):
    st.subheader("Model check: does transfer learning help?")
    info_panel(
        "Shows, for the yield outlook, how well it predicts WITHOUT transfer learning, then applies transfer learning and shows the "
        "improvement (or the lack of one). Also shows the stored TrAdaBoost results for the NDVI forecast and for yield.",
        "",
        "Decide whether the more complex method is worth using, based on measured errors and not on the name of the method.",
        "back-tests use 40 granary-years (10 granaries, 2021 to 2024) for yield, so small differences are noise. Look at the confidence "
        "interval, not only the average.")
    with st.expander("How the backend works: what is trained and what is not"):
        st.markdown(
            "**1. NDVI forecast (10 and 30 days ahead): a trained model.** It learns from the 10-day satellite and weather series of all "
            "granaries together, year by year (train on earlier years, test on the next). Ridge regression and gradient boosting are "
            "compared with two baselines, 'no change' and a seasonal average, and the best is chosen on 2021 to 2023, keeping 2024 untouched. "
            "It drives crop status, the stage estimate and resilience.\n\n"
            "**2. Yield outlook (1 to 3 years ahead): transfer learning of yield behaviour.** One granary has about 7 yearly yields, too few "
            "to learn from. The other granaries show how much one year's yield carries into the next, so that carry-over rate is learned on "
            "them and applied to each granary. TrAdaBoost (re-weighting other granaries' rows) was tested on NDVI and on yield and did not "
            "help, so it is not used.\n\n"
            "**3. Advisory and intervention: not a trained model.** Editable rules turn the results above into actions (who, when, why), each "
            "tied to a source. An AI model, when a key is set on the server, only writes the wording from that evidence.")
    g = selected_granary()
    yh = D.yh[["District", "Year", "Yield"]].dropna()
    applied = st.session_state.get("tl_applied", False)

    # ------------------------------------------------------------------ yield: before
    st.markdown("### Yield outlook")
    st.markdown("**Step 1. Without transfer learning.** The outlook is learned from each granary's own few yearly values. The reference "
                "is 'same as last year', which needs no model.")
    before = _backtest(yh, False)
    names0 = {"last_year": "Same as last year (no model), error %", "ewma_own": "Normal training (one granary alone), error %"}
    t0 = _table(before, ["last_year", "ewma_own"], names0)
    st.dataframe(t0, width="stretch")
    own = _own_outlook(D.yh[D.yh.District == g].sort_values("Year"), before)
    last_year = int(D.yh[D.yh.District == g].Year.max())
    st.caption(f"Outlook for {g} without transfer learning (central value, typical error band):  " + "   |   ".join(
        f"{last_year + h}: {own[h][0]:,.0f} kg/ha (±{own[h][1]:.0f}%)" for h in (1, 2, 3)))

    # ------------------------------------------------------------------ the button
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Apply transfer learning", type="primary", key="tl_apply"):
        st.session_state["tl_applied"] = True
        st.rerun()
    if applied and b2.button("Reset (show without)", key="tl_reset"):
        st.session_state["tl_applied"] = False
        st.rerun()
    if not applied:
        st.caption("Press the button to apply transfer learning to the same back-test and compare.")
        return

    # ------------------------------------------------------------------ yield: after
    with st.spinner("Applying transfer learning and repeating the back-test..."):
        after = _backtest(yh, True)
    st.markdown("**Step 2. With transfer learning.** The carry-over rate is learned on the other granaries and applied to each granary, "
                "and the same back-test is repeated.")
    names1 = {**names0, "ar1_transfer": "Transfer learning, error %"}
    t1 = _table(after, ["last_year", "ewma_own", "ar1_transfer"], names1)
    rows = []
    for h, grp in after.groupby("horizon"):
        w = grp.pivot_table(index=["test_year", "District"], columns="method", values="ape")
        d = (w["ar1_transfer"] - w["ewma_own"]).groupby(level=1).mean()
        lo, hi = boot(d.values)
        d2 = (w["ar1_transfer"] - w["last_year"]).groupby(level=1).mean()
        lo2, hi2 = boot(d2.values)
        rows.append({"Outlook": HLAB[h], "Change vs normal training (points)": round(d.mean(), 2),
                     "95% range": f"{lo:+.2f} to {hi:+.2f}", "Better than normal training in": f"{(w['ar1_transfer'] < w['ewma_own']).mean():.0%} of cases",
                     "Result": _verdict(lo, hi), "Change vs same as last year (points)": round(d2.mean(), 2),
                     "95% range ": f"{lo2:+.2f} to {hi2:+.2f}", "Result ": _verdict(lo2, hi2)})
    st.dataframe(t1, width="stretch")
    st.dataframe(pd.DataFrame(rows).set_index("Outlook"), width="stretch")
    fig = go.Figure()
    for col, nm, c in (("last_year", "Same as last year (no model)", GREY), ("ewma_own", "Normal training", BLUE),
                       ("ar1_transfer", "Transfer learning", GREEN)):
        fig.add_bar(x=list(t1.index), y=t1[names1[col]], name=nm, marker_color=c,
                    text=[f"{v:.1f}%" for v in t1[names1[col]]], textposition="outside")
    fig.update_layout(height=330, barmode="group", yaxis_title="Average error, % (lower is better)", margin=dict(t=20, b=30),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch", key="tl_bar")
    tl = _tl_outlook(D, g, after)
    st.markdown(f"**{g}: outlook before and after**")
    st.dataframe(pd.DataFrame([{"Year": last_year + h, "Without transfer learning (kg/ha)": f"{own[h][0]:,.0f}  (±{own[h][1]:.0f}%)",
                                "With transfer learning (kg/ha)": f"{tl[h][0]:,.0f}  (±{tl[h][1]:.0f}%)",
                                "Difference (kg/ha)": f"{tl[h][0] - own[h][0]:+,.0f}", "Carry-over rate learned": f"{tl[h][2]:.2f}"}
                               for h in (1, 2, 3)]).set_index("Year"), width="stretch")
    st.caption("Carry-over rate: close to 1 means next year's yield stays near last year's; near 0.5 means it moves half way back to the "
               "granary's usual level. It is learned from the other granaries, not from this one.")

    # ------------------------------------------------------------------ TrAdaBoost (stored results)
    st.markdown("### TrAdaBoost (re-weights other granaries' rows)")
    st.caption("Too slow to repeat here, so these are the stored results of the last pipeline run "
               "(pipeline/transfer_benchmark.py and pipeline/yield_transfer_benchmark.py). Same test for every method.")
    verdict = []
    ts = getattr(D, "tl_summary", None)
    if ts is not None and not ts.empty:
        k0 = ts.k_years.min()
        s = ts[ts.k_years == k0].copy()
        s["Forecast"] = s.horizon.map({1: "NDVI, 10 days ahead", 3: "NDVI, 30 days ahead"})
        s["Best normal training"] = s[["pooled_ridge", "pooled_hgb", "pooled_ada"]].min(axis=1)
        s["Change with TrAdaBoost (%)"] = ((s.tradaboost - s["Best normal training"]) / s["Best normal training"] * 100).round(0)
        s["Result"] = np.where(s["Change with TrAdaBoost (%)"] < -5, "Improved", np.where(s["Change with TrAdaBoost (%)"] > 5, "Worse", "No clear difference"))
        st.markdown(f"**NDVI forecast** (a granary with {int(k0)} year of its own data; average error in NDVI units, lower is better)")
        st.dataframe(s[["Forecast", "persistence", "Best normal training", "tradaboost", "Change with TrAdaBoost (%)", "Result"]]
                     .rename(columns={"persistence": "No change", "tradaboost": "TrAdaBoost"}).set_index("Forecast"), width="stretch")
        r1 = s[s.horizon == 1].iloc[0]
        verdict.append(("TrAdaBoost", "NDVI forecast, 10 days", f"{r1['Best normal training']:.4f}", f"{r1.tradaboost:.4f}", r1["Result"]))
    yt = getattr(D, "yield_tab", None)
    if yt is not None and not yt.empty:
        r2 = yt[yt.k_years == yt.k_years.min()].iloc[0]
        st.markdown(f"**Yield from satellite and weather inputs** (a granary with {int(r2.k_years)} years of yield; average error %)")
        st.dataframe(yt.rename(columns={"k_years": "Years of own yield", "last_year": "Last year", "own_mean": "Own average",
                                        "avg_last3": "Last 3 years", "target_only": "One granary alone", "pooled_ridge": "Normal pooled",
                                        "tradaboost": "TrAdaBoost"}).set_index("Years of own yield"), width="stretch")
        d = r2.tradaboost - r2.own_mean
        verdict.append(("TrAdaBoost", "Yield from inputs", f"{r2.own_mean:.1f}%", f"{r2.tradaboost:.1f}%",
                        "Improved" if d < -0.5 else ("Worse" if d > 0.5 else "No clear difference")))

    iw = getattr(D, "yield_iw", None)
    if iw is not None and not iw.empty:
        st.markdown("**Instance weighting applied to the yield dynamics** (year-to-year transitions as the instances; average error %, "
                    "every method on the same cases)")
        t = iw.copy()
        t["Outlook"] = t.horizon.map(HLAB)
        cols = {"cases": "Cases", "last_year": "Same as last year", "ar1_transfer": "Transfer (equal weights)",
                "ar1_similarity": "Instance weighting: similar granaries count more", "pool_upweight": "Instance weighting: own years x5",
                "tradaboost_ridge": "TrAdaBoost (ridge)", "tradaboost_stump": "TrAdaBoost (stumps)"}
        st.dataframe(t.set_index("Outlook")[list(cols)].rename(columns=cols), width="stretch")
        st.caption("Three years ahead is not shown: a granary has no transition of its own at that distance, so instance weighting cannot start.")
        r1 = t[t.horizon == 1].iloc[0]
        best = min(("ar1_similarity", "pool_upweight", "tradaboost_ridge", "tradaboost_stump"), key=lambda c: r1[c])
        dd = r1[best] - r1.ar1_transfer
        verdict.append(("Instance weighting (similarity, up-weighting, TrAdaBoost)", "Yield outlook, 1 year", f"{r1.ar1_transfer:.1f}%",
                        f"{r1[best]:.1f}%", "Improved" if dd < -0.5 else ("Worse" if dd > 0.5 else "No clear difference")))

    # ------------------------------------------------------------------ the answer
    h1 = after[after.horizon == 1].pivot_table(index=["test_year", "District"], columns="method", values="ape")
    d1 = (h1["ar1_transfer"] - h1["ewma_own"]).groupby(level=1).mean()
    lo, hi = boot(d1.values)
    verdict.insert(0, ("Transfer of yield behaviour", "Yield outlook, 1 year", f"{h1['ewma_own'].mean():.1f}%", f"{h1['ar1_transfer'].mean():.1f}%",
                       _verdict(lo, hi)))
    st.markdown("### The answer")
    st.dataframe(pd.DataFrame(verdict, columns=["Method", "Task", "Error without", "Error with", "Result"]).set_index("Method"), width="stretch")
    gain = h1["ewma_own"].mean() - h1["ar1_transfer"].mean()
    vs_last = (h1["ar1_transfer"] - h1["last_year"]).mean()
    st.success(f"Transfer of yield behaviour lowers the 1-year yield error from {h1['ewma_own'].mean():.1f}% to {h1['ar1_transfer'].mean():.1f}% "
               f"({gain:.1f} points better than normal training). Against simply using last year's yield it is "
               f"{'about the same' if abs(vs_last) < 0.5 else ('better' if vs_last < 0 else 'worse')} ({vs_last:+.1f} points), so its value is that "
               "it learns this from the data instead of assuming it. TrAdaBoost and the other instance-weighting methods did not improve any task.")
    st.caption("These are back-tests on 10 granaries and 7 years of yield. Treat them as indications, and re-run the pipeline when more years arrive.")
