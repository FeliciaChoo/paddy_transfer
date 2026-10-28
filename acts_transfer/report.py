"""One downloadable report for the whole workflow: assessment, harmonisation, suitability, results, diagnostics, and exactly how it was run."""
import json

import pandas as pd

from ._util import md_table
from .methods import LABELS


def workflow_markdown(wf):
    L = [f"# Transfer-learning assessment and evaluation: {wf.source.name} to {wf.target.name}", ""]
    L += ["## Workflow status", md_table(pd.DataFrame(wf.status())[["step", "status", "detail"]]), ""]
    if wf.report is not None:
        L += [wf.report.to_markdown().replace("\n# ", "\n## ").replace("# Pre-transfer assessment", "## Pre-transfer assessment", 1), ""]
    if wf.bundle is not None:
        b = wf.bundle
        L += ["## Data harmonisation", f"Fingerprint `{b.token}`; originals are unchanged.", "", "### Transformations applied",
              md_table(pd.DataFrame([{**l, "detail": str(l["detail"])} for l in b.log])), "", "### Features", md_table(b.feature_table), "",
              "### Validation", md_table(pd.DataFrame(b.validation)), ""]
    if wf.suit is not None:
        s = wf.suit
        L += ["## Transfer suitability", f"**{s['suitability'].replace('_', ' ').upper()}**", "", *[f"- {r}" for r in s["reasons"]], "", "### Scenarios A to F"]
        for sc in s["scenarios"]:
            L += [f"**{sc['id']}. {sc['name']}**: {'CONDITIONS MET' if sc['met'] else 'not met'}", f"- Observed: {sc['observed']}", f"- Evidence: {sc['evidence']}",
                  f"- Recommended: {', '.join(LABELS.get(m, m) for m in sc['methods']) or 'none'}; runnable here: {', '.join(sc['executable']) or 'none'}",
                  f"- Why: {sc['why']}", f"- Assumptions: {sc['assumptions']}", f"- Benefits and risks: {sc['benefits_risks']}", f"- Alternatives: {sc['alternatives']}", ""]
        L += ["### Method table", md_table(s["methods"][["method", "family", "eligible", "reason", "recommended_by", "features_compared", "scaling", "metric",
                                                         "distance_influence", "limitations"]]), ""]
    if getattr(wf, "override", None):
        L += [f"**Override of a NOT RECOMMENDED verdict. Reason recorded:** {wf.override}", ""]
    if getattr(wf, "results", None):
        L += ["## Transfer performance comparison (yield in the harmonised unit, held-out seasons only)"]
        for strat, r in wf.results.items():
            L += [f"### Validation: {strat} ({r.n_folds} held-out seasons)", md_table(r.metrics[["method", "n_test", "MAE", "RMSE", "R2", "R2_note"]]), "",
                  "Change in mean absolute error against each baseline (negative is better):", md_table(r.improvement), "",
                  "Absolute error by season:", md_table(r.per_group), ""]
            L += [f"- {n}" for n in r.notes] + [""]
        if wf.result.failures:
            L += ["### Methods that failed", md_table(pd.DataFrame(wf.result.failures)), ""]
        L += ["## Diagnostics"]
        for m in [k for k in wf.result.methods if k not in ("source_only", "target_only", "target_mean", "pooled")]:
            for base in ("target_only", "season_trend"):
                d, summ = wf.diagnose(m, base)
                if len(d):
                    L += [f"### {LABELS.get(m, m)} versus {LABELS.get(base, base)}", summ, "", md_table(d), ""]
    L += ["## Settings", "```", json.dumps(wf.cfg.to_dict(), indent=1, default=str), "```", f"Base model: {wf.base}", "", "## Event log"]
    L += [f"- {t}: {m}" for t, m in wf.events]
    return "\n".join(L)
