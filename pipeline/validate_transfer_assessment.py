"""
validate_transfer_assessment.py -- does the pre-transfer assessment PREDICT where transfer helps or hurts?

    python pipeline/validate_transfer_assessment.py --source "IADA Barat Laut Selangor" --out data

For one source granary and every other granary as the target: run the full workflow (assess, harmonise, suitability, a fixed set of transfer methods,
leave-one-season-out and forward validation) and relate the assessment's measures to the change in prediction error. Each target has only 7 yearly
yields, so every row is noisy, and with 9 targets any correlation is only an indication.

Per target it records  climate AUC, vegetation/radar AUC, ten-day MMD^2, calendar offset, label shift, the suitability verdict, and the change in mean
absolute error of the transfer methods against the target-only model and against the pure season-trend baseline (forward validation).
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acts_transfer import Config, Domain, TransferWorkflow          # noqa: E402
from acts_transfer.assessment import mmd2_unbiased                   # noqa: E402
warnings.filterwarnings("ignore")

FEATS = ["NDVI_mean", "NDVI_max", "NDVI_amplitude", "EVI_mean", "SAR_VV_dB_mean", "Mean_Temp_C_mean", "Annual_Rain_mm", "Solar_Radiation_MJ_mean"]
FINE = ["NDVI", "EVI", "SAR_VV_dB", "Mean_Temp_C", "Rain_decade_mm", "Solar_Radiation_MJ"]
UNITS = {"NDVI_mean": "unitless", "NDVI_max": "unitless", "NDVI_amplitude": "unitless", "EVI_mean": "unitless", "SAR_VV_dB_mean": "dB", "Mean_Temp_C_mean": "deg C",
         "Annual_Rain_mm": "mm per year", "Solar_Radiation_MJ_mean": "MJ/m2", "Yield_kg_ha": "kg/ha"}
METHODS = ["tradaboost_r2", "kmm", "coral", "tca", "ridge_prior"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default="IADA Barat Laut Selangor")
    ap.add_argument("--yield-data", default="data/transfer_yield_dataset.csv")
    ap.add_argument("--fine-data", default="data/transfer_decade_features.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--permutations", type=int, default=100)
    a = ap.parse_args()
    Y, F = pd.read_csv(a.yield_data), pd.read_csv(a.fine_data)
    dom = lambda g: Domain(g, Y[Y.District == g].sort_values("Year").reset_index(drop=True), list(FEATS), "Yield_kg_ha", "Year", dict(UNITS),
                           "Optical_observed_pct", F[F.District == g].reset_index(drop=True), list(FINE))
    rows = []
    for t in sorted(Y.District.unique()):
        if t == a.source:
            continue
        wf = TransferWorkflow(Config(mmd_permutations=a.permutations))
        wf.select_domains(dom(a.source), dom(t), "ridge")
        rep = wf.assess()
        wf.harmonise()
        su = wf.evaluate_suitability()
        ok = [m for m in METHODS if bool(su["methods"].set_index("key").loc[m, "eligible"])]
        wf.select_strategy(ok)
        wf.run(override_reason="validation of the assessment" if su["suitability"] == "not_recommended" else None)
        fd = rep.fine_diag or {}
        zs = pd.concat([wf.source.fine_df[FINE], wf.target.fine_df[FINE]]).dropna()
        zs = (zs - zs.mean()) / zs.std()
        n1 = len(wf.source.fine_df[FINE].dropna())
        row = dict(target=t, suitability=su["suitability"], readiness=rep.readiness, climate_auc=fd.get("auc_climate"), vegetation_auc=fd.get("auc_vegetation"),
                   calendar_offset_days=fd.get("calendar_offset_days"), mmd2_tenday=mmd2_unbiased(zs.values[:n1], zs.values[n1:])[0],
                   label_smd=rep.label["SMD"] if rep.label else np.nan)
        for strat, r in wf.results.items():
            wide = r.predictions.pivot_table(index="group", columns="method", values="abs_err")
            mae = wide.mean()
            for ref in ("target_only", "season_trend"):
                if ref in mae:
                    ch = [100 * (mae[m] / mae[ref] - 1) for m in ok if m in mae]
                    row[f"{strat}_mean_change_vs_{ref}_pct"] = float(np.mean(ch))
                    row[f"{strat}_median_change_vs_{ref}_pct"] = float(np.median(ch))
                    row[f"{strat}_best_change_vs_{ref}_pct"] = float(np.min(ch))
            row[f"{strat}_MAE_target_only"] = float(mae.get("target_only", np.nan))
            row[f"{strat}_MAE_season_trend"] = float(mae.get("season_trend", np.nan))
            row[f"{strat}_MAE_source_only"] = float(mae.get("source_only", np.nan))
        rows.append(row)
        print("done", t)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(os.path.join(a.out, "transfer_assessment_validation.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df[["target", "suitability", "climate_auc", "vegetation_auc", "calendar_offset_days", "mmd2_tenday", "label_smd", "forward_mean_change_vs_target_only_pct",
              "forward_median_change_vs_target_only_pct", "forward_best_change_vs_target_only_pct", "forward_median_change_vs_season_trend_pct"]].to_string(index=False))
    print("\nSpearman correlation of each assessment measure with the MEDIAN change in error of the transfer methods against target-only (forward validation; n =", len(df), "targets):")
    for c in ("climate_auc", "vegetation_auc", "mmd2_tenday", "calendar_offset_days", "label_smd"):
        d = df[[c, "forward_median_change_vs_target_only_pct"]].dropna()
        if len(d) >= 5:
            rho, p = spearmanr(d[c], d["forward_median_change_vs_target_only_pct"])
            print(f"  {c:22s} rho = {rho:+.2f}  p = {p:.3f}")


if __name__ == "__main__":
    main()
