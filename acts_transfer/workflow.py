"""The seven-step workflow and its gates, enforced HERE (the backend). An interface can hide a button, but it cannot skip a step.

  Step 1 select domains -> 2 assess data -> 3 harmonise -> 4 evaluate suitability -> 5 select strategy -> 6 run transfer -> 7 compare results

How the order is enforced:
  * every artifact carries a fingerprint (SHA-256) of the exact data and settings it was built from;
  * harmonise() recomputes the fingerprint of the current source and target data and refuses a report that no longer matches (StaleAssessment);
  * run() requires, in this order: a valid assessment, a harmonised bundle built from THAT assessment, a suitability result built from THAT
    bundle, and eligible methods. If the suitability is NOT RECOMMENDED, run() also needs an explicit override reason, which is written into the
    result and the report;
  * changing the domains or the settings clears every later step.
There is no function that trains a transfer method on raw data: evaluation.evaluate() takes only a HarmonisedBundle, which only harmonise()
can create (and only from a non-blocked assessment).
"""
import dataclasses
from datetime import datetime

import pandas as pd

from . import assessment, diagnostics, evaluation, harmonise, selection
from .errors import WorkflowError
from .config import Config, Domain
from .evaluation import BASELINES
from .methods import REGISTRY

STEPS = [("select", "1 · Select domains"), ("assess", "2 · Assess data"), ("harmonise", "3 · Harmonise data"),
         ("suitability", "4 · Evaluate suitability"), ("strategy", "5 · Select strategy"), ("run", "6 · Run transfer"), ("compare", "7 · Compare results")]


class TransferWorkflow:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.source = self.target = None
        self.base = "ridge"
        self._clear_from("report")
        self.events = []

    # ------------------------------------------------------------------ state
    ORDER = ["report", "bundle", "suit", "strategy", "result", "results", "override"]

    def _clear_from(self, name):
        """drop this artifact and every later one, so nothing stale can survive a change upstream"""
        for n in self.ORDER[self.ORDER.index(name):]:
            setattr(self, n, None)

    def _log(self, msg):
        self.events.append((datetime.now().strftime("%H:%M:%S"), msg))

    def status(self):
        """one row per step: complete / current / locked / blocked, with the reason"""
        done = [self.source is not None, self.report is not None, self.bundle is not None, self.suit is not None, self.strategy is not None,
                self.result is not None, self.result is not None]
        rows, current_set = [], False
        for i, (key, label) in enumerate(STEPS):
            if done[i]:
                st, why = "complete", ""
            elif all(done[:i]) and not current_set:
                st, why = "current", ""
                current_set = True
            else:
                st, why = "locked", "complete the previous step first"
            if key == "harmonise" and self.report is not None and self.report.readiness == "not_ready":
                st, why = "blocked", "data readiness is NOT READY: " + "; ".join(self.report.blockers)
            if key == "suitability" and self.suit is not None and self.suit["suitability"] == "not_recommended":
                st, why = "complete", "transfer is NOT RECOMMENDED"
            rows.append(dict(key=key, step=label, status=st, detail=why))
        return rows

    # ------------------------------------------------------------------ steps
    def select_domains(self, source: Domain, target: Domain, base="ridge", cfg: Config = None):
        if source.name == target.name:
            raise WorkflowError("source and target must be different domains")
        if cfg is not None:
            self.cfg = cfg
        self.source, self.target, self.base = source, target, base
        self._clear_from("report")
        self._log(f"selected source '{source.name}' and target '{target.name}', base model {base}")

    def assess(self):
        if self.source is None:
            raise WorkflowError("Step 1 (select domains) is not complete")
        self._clear_from("report")
        self.report = assessment.assess(self.source, self.target, self.cfg)
        self._log(f"assessment: readiness {self.report.readiness}")
        return self.report

    def harmonise(self, spec=None):
        if self.report is None:
            raise WorkflowError("Step 2 (assess data) must be completed before harmonisation")
        spec = spec or harmonise.default_spec(self.report, self.source, self.target)
        self._clear_from("bundle")
        self.bundle = harmonise.harmonise(self.report, self.source, self.target, spec, self.cfg)      # raises Stale/Blocked
        self._log(f"harmonisation: {'validated' if self.bundle.ready else 'validation FAILED'}")
        if not self.bundle.ready:
            raise WorkflowError("harmonisation validation failed: " + "; ".join(v["check"] for v in self.bundle.validation if not v["passed"]))
        return self.bundle

    def evaluate_suitability(self):
        self._require_chain(upto="harmonise")
        self._clear_from("suit")
        self.suit = selection.select(self.report, self.bundle, self.cfg, self.base)
        self._log(f"suitability: {self.suit['suitability']}")
        return self.suit

    def select_strategy(self, methods):
        self._require_chain(upto="suitability")
        elig = self.suit["methods"].set_index("key")
        bad = [m for m in methods if m not in elig.index or not bool(elig.loc[m, "eligible"])]
        if bad:
            raise WorkflowError("not eligible: " + "; ".join(f"{m} ({elig.loc[m, 'reason'] if m in elig.index else 'unknown'})" for m in bad))
        self._clear_from("strategy")
        self.strategy = list(dict.fromkeys(list(methods)))
        self._log(f"strategy selected: {self.strategy}")
        return self.strategy

    def run(self, methods=None, override_reason=None):
        """Step 6. Needs every earlier step, a matching fingerprint chain, eligible methods, and (if not recommended) an override reason."""
        self._require_chain(upto="suitability")
        methods = list(methods if methods is not None else (self.strategy or []))
        if not methods:
            raise WorkflowError("Step 5 (select strategy) is not complete: no method was selected")
        if self.suit["suitability"] == "not_recommended" and not (override_reason and override_reason.strip()):
            raise WorkflowError("transfer is NOT RECOMMENDED for these domains: running needs an explicit override reason, which is recorded in the report")
        elig = self.suit["methods"].set_index("key")
        bad = [m for m in methods if m not in elig.index or not bool(elig.loc[m, "eligible"])]
        if bad:
            raise WorkflowError("not eligible: " + ", ".join(bad))
        full = [b for b in BASELINES if b in elig.index and bool(elig.loc[b, "eligible"])] + [m for m in methods if m not in BASELINES]
        self._clear_from("result")
        self.override = override_reason.strip() if override_reason else None
        full = list(dict.fromkeys(full))
        # both validation strategies are always run: leave-one-season-out and the stricter forward (earlier seasons only) check
        self.results = {s: evaluation.evaluate(self.bundle, full, self.base, dataclasses.replace(self.cfg, cv_strategy=s)) for s in ("loso", "forward")}
        self.result = self.results[self.cfg.cv_strategy]
        self.result.notes.append("Baselines (source-only, target-only, target mean, pooled) and a season-trend baseline are always included, so every method is judged against them.")
        if self.override:
            self.result.notes.append(f"Run with an explicit override of a NOT RECOMMENDED verdict. Reason given: {self.override}")
        self._log(f"ran {full}")
        return self.result

    def diagnose(self, method, baseline):
        self._require_result()
        return diagnostics.diagnose(self.bundle, self.report, self.result, method, baseline, self.cfg)

    # ------------------------------------------------------------------ gates
    def _require_chain(self, upto):
        if self.report is None:
            raise WorkflowError("Step 2 (assess data) is not complete")
        if self.report.token != assessment.token_of(self.source, self.target, self.cfg):
            raise harmonise.StaleAssessment("the data or settings changed after the assessment; assess again")
        if self.report.readiness == "not_ready":
            raise harmonise.AssessmentBlocked("data readiness is NOT READY: " + "; ".join(self.report.blockers))
        if self.bundle is None:
            raise WorkflowError("Step 3 (harmonise data) is not complete")
        if self.bundle.assessment_token != self.report.token or not self.bundle.ready:
            raise WorkflowError("the harmonised data does not belong to the current assessment, or failed validation")
        if upto in ("suitability",):
            if self.suit is None:
                raise WorkflowError("Step 4 (evaluate suitability) is not complete")
            if self.suit["bundle_token"] != self.bundle.token:
                raise WorkflowError("the suitability result belongs to different harmonised data")

    def _require_result(self):
        if self.result is None:
            raise WorkflowError("Step 6 (run transfer) is not complete")

    def can_run(self):
        try:
            self._require_chain(upto="suitability")
            return True, ""
        except Exception as e:
            return False, str(e)
