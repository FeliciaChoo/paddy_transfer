"""Decision tracker: what KPKM decided about each proposed intervention, and what happened next.

Every save APPENDS rows to the database (dashboard/store.py: SQLite file data/acts.db by default, or the DATABASE_URL you set); nothing is overwritten, so the file is a history. The CURRENT state of an item is
its latest row. An item is identified by  granary | rule id | data-as-of date,  so a decision belongs to the advice that was given
for that data; when the data is refreshed and new advice appears, it starts as Pending again instead of inheriting an old decision.

Decisions   Accept (do it as proposed), Modify (do something different: say what), Reject (do not do it: say why), Defer (not now).
Status      Not started, In progress, Done, Dropped (a rejected item is Dropped by default).
The special row 'OVERALL' holds KPKM's overall next step / conclusion for a granary.
"""
from datetime import date, datetime

import pandas as pd

from . import store
from .store import COLS

DECISIONS = ["Pending", "Accept", "Modify", "Reject", "Defer"]
STATUSES = ["Not started", "In progress", "Done", "Dropped"]
EDIT_COLS = ["decision", "kpkm_action", "responsible", "target_date", "status", "reason"]
OVERALL = "OVERALL"


def load(data_dir):
    """the whole history, from the database (see store.py)"""
    return store.read(data_dir)


def current(df):
    """latest row per item"""
    if df.empty:
        return df
    return df.sort_values("saved_at", kind="stable").groupby("key", sort=False).tail(1).reset_index(drop=True)


def make_key(case, rule_id):
    return f"{case['granary']}|{rule_id}|{case.get('as_of', '')}"


def default_status(decision):
    return "Dropped" if decision == "Reject" else "Not started"


def _date(v):
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) in ("", "NaT", "None", "nan"):
        return ""
    return str(v)[:10]


def _as_date(s):
    s = _date(s)
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def editor_frame(case, acts, cur):
    """the table KPKM edits: one row per proposed intervention, prefilled with any saved decision"""
    from . import bridge                                   # local import: bridge does not depend on the tracker
    have = {r.key: r for r in cur.itertuples()} if not cur.empty else {}
    rows = []
    for a in acts:
        rec = have.get(make_key(case, a[4]))
        rows.append({"Rule": a[4], "Priority": bridge.priority_label(a[7]), "Proposed intervention": a[0],
                     "Decision": rec.decision if rec else "Pending",
                     "What KPKM will do (next step)": rec.kpkm_action if rec else "",
                     "Responsible officer": rec.responsible if rec else "",
                     "Target date": _as_date(rec.target_date) if rec else None,
                     "Status": rec.status if rec and rec.status else "Not started",
                     "Reason / remarks": rec.reason if rec else ""})
    return pd.DataFrame(rows)


def overall_text(case, cur):
    if cur.empty:
        return ""
    r = cur[cur.key == make_key(case, OVERALL)]
    return r.kpkm_action.iloc[0] if len(r) else ""


def validate(frame):
    """messages for rows that cannot be saved as they are"""
    msgs = []
    for r in frame.itertuples(index=False):
        d, act, why = r[3], str(r[4] or "").strip(), str(r[8] or "").strip()
        name = str(r[2])[:60] + ("..." if len(str(r[2])) > 60 else "")
        if d == "Modify" and not act:
            msgs.append(f"Modify: say what KPKM will do instead ({name})")
        if d == "Reject" and not why:
            msgs.append(f"Reject: give the reason ({name})")
    return msgs


def _record(case, rule_id, priority, proposed, decision, action, who, when, status, reason, by):
    return dict(saved_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), key=make_key(case, rule_id), granary=case["granary"],
                rule_id=rule_id, priority=priority, proposed=proposed, decision=decision, kpkm_action=action or "",
                responsible=who or "", target_date=_date(when), status=status or default_status(decision), reason=reason or "",
                decided_by=by or "", data_as_of=case.get("as_of", ""), crop_status=case.get("status", ""), confidence=case["confidence"])


def _norm(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) in ("None", "nan", "NaT") else str(v)


def _same(old, new):
    return all(_norm(old[c]) == _norm(new[c]) for c in EDIT_COLS)


def save(data_dir, case, acts, frame, by, overall):
    """append a row for every item that is new or changed; returns how many rows were written"""
    hist = load(data_dir)
    cur = current(hist)
    have = {r["key"]: r for _, r in cur.iterrows()}
    new = []
    for a, r in zip(acts, frame.itertuples(index=False)):
        dec, action, who, when, status, why = r[3], r[4], r[5], r[6], r[7], r[8]
        rec = _record(case, a[4], a[7], a[0], dec, action, who, when, status, why, by)
        old = have.get(rec["key"])
        if old is None and dec == "Pending":
            continue                                       # nothing decided, nothing to keep
        if old is None or not _same(old, rec) or (by and old["decided_by"] != by):
            new.append(rec)
    ov = _record(case, OVERALL, 0, "(overall next step and conclusion for this granary)", "Noted", overall.strip(), "", None, "Not started",
                 "", by)
    old = have.get(ov["key"])
    if overall.strip() and (old is None or old["kpkm_action"] != overall.strip()):
        new.append(ov)
    if new:
        store.append(data_dir, pd.DataFrame(new))
    return len(new)


def save_edits(data_dir, original, edited, by):
    """tracker tab: write new versions of the items whose follow-up fields were edited (history is kept)"""
    hist = load(data_dir)
    new = []
    for (_, o), (_, e) in zip(original.iterrows(), edited.iterrows()):
        changed = any(_date(o[c]) != _date(e[c]) if c == "target_date" else _norm(o[c]) != _norm(e[c]) for c in EDIT_COLS)
        if changed:
            r = o.to_dict()
            r.update({c: (_date(e[c]) if c == "target_date" else _norm(e[c])) for c in EDIT_COLS})
            r["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if by:
                r["decided_by"] = by
            new.append(r)
    if new:
        store.append(data_dir, pd.DataFrame(new))
    return len(new)


def recorded_for(cur, granary, as_of):
    """{rule_id: row} of the saved decisions for one granary and data date (the report prefills from this)"""
    if cur.empty:
        return {}
    r = cur[(cur.granary == granary) & (cur.data_as_of == as_of)]
    return {x.rule_id: x for x in r.itertuples()}


def overdue(cur, today=None):
    today = (today or date.today()).isoformat()
    if cur.empty:
        return cur
    m = (cur.target_date != "") & (cur.target_date < today) & ~cur.status.isin(["Done", "Dropped"]) & (cur.rule_id != OVERALL)
    return cur[m]


def summary(cur):
    items = cur[cur.rule_id != OVERALL] if not cur.empty else cur
    out = {d: int((items.decision == d).sum()) for d in DECISIONS if d != "Pending"} if not items.empty else {d: 0 for d in DECISIONS[1:]}
    out["recorded"] = int(sum(out.values()))
    out["open"] = int((~items.status.isin(["Done", "Dropped"]) & (items.decision.isin(["Accept", "Modify", "Defer"]))).sum()) if not items.empty else 0
    out["overdue"] = len(overdue(cur)) if not items.empty else 0
    return out


def by_rule(cur, rules):
    """how KPKM's decisions differ from the advice, per rule: counts of accept / modify / reject / defer"""
    items = cur[(cur.rule_id != OVERALL) & (cur.decision != "Pending")] if not cur.empty else cur
    if items.empty:
        return pd.DataFrame()
    t = items.pivot_table(index="rule_id", columns="decision", values="key", aggfunc="count", fill_value=0)
    for d in DECISIONS[1:]:
        if d not in t.columns:
            t[d] = 0
    t = t[DECISIONS[1:]]
    t["Total"] = t.sum(axis=1)
    t["Changed or rejected %"] = ((t["Modify"] + t["Reject"]) / t["Total"] * 100).round(0)
    lab = rules.set_index("rule_id").action.str.replace(r"\{[^}]+\}", "...", regex=True).str.slice(0, 70)
    t.insert(0, "Intervention (rule)", [lab.get(i, "") for i in t.index])
    return t.reset_index().rename(columns={"rule_id": "Rule"}).sort_values("Total", ascending=False)
