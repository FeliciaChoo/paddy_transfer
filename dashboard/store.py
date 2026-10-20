"""Storage for the decision tracker: a real database, with no setup by default.

Which database
  default     SQLite, one file: data/acts.db  (built into Python; PyCharm can open it, see README)
  any other   set DATABASE_URL (environment variable, or .streamlit/secrets.toml) to a SQLAlchemy URL, for example
                  postgresql+psycopg2://user:password@host:5432/dbname       (pip install psycopg2-binary)
                  mysql+pymysql://user:password@host:3306/dbname             (pip install pymysql)
                  mssql+pyodbc://...                                         (SQL Server)
              The table is created on first use. Nothing else in the app changes.
  fallback    if SQLAlchemy is not installed, a CSV file (data/decision_tracker.csv) is used, as before.

Rows are only ever APPENDED (the table is a history; the tracker reads the latest row per item). A CSV left over from an earlier
version is imported once, the first time the database is empty.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import sqlalchemy as sa
except ImportError:                                        # CSV fallback
    sa = None

TABLE = "decision_tracker"
COLS = ["saved_at", "key", "granary", "rule_id", "priority", "proposed", "decision", "kpkm_action", "responsible", "target_date",
        "status", "reason", "decided_by", "data_as_of", "crop_status", "confidence"]


def _cfg(name):
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:                                      # no secrets file
        return ""


def database_url(data_dir):
    return _cfg("DATABASE_URL") or "sqlite:///" + (Path(data_dir) / "acts.db").resolve().as_posix()


def _csv(data_dir):
    return Path(data_dir) / "decision_tracker.csv"


def _table(meta):
    return sa.Table(TABLE, meta, sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    *[sa.Column(c, sa.Text) for c in COLS])


@st.cache_resource(show_spinner=False)
def _engine(url, csv_path):
    """one engine per database URL; creates the table, and imports an old CSV into an empty table once"""
    eng = sa.create_engine(url, pool_pre_ping=True)
    meta = sa.MetaData()
    t = _table(meta)
    meta.create_all(eng)
    with eng.begin() as c:
        empty = c.execute(sa.select(sa.func.count()).select_from(t)).scalar() == 0
        if empty and Path(csv_path).exists():
            old = pd.read_csv(csv_path, dtype=str).fillna("")
            for col in COLS:
                if col not in old.columns:
                    old[col] = ""
            if len(old):
                c.execute(t.insert(), old[COLS].to_dict("records"))
    return eng


def read(data_dir):
    """the whole history as strings, oldest first"""
    if sa is None:
        p = _csv(data_dir)
        if not p.exists():
            return pd.DataFrame(columns=COLS)
        return pd.read_csv(p, dtype=str).fillna("").reindex(columns=COLS, fill_value="")
    eng = _engine(database_url(data_dir), str(_csv(data_dir)))
    t = _table(sa.MetaData())
    with eng.connect() as c:
        rows = c.execute(sa.select(*[t.c[x] for x in COLS]).order_by(t.c.saved_at, t.c.id)).fetchall()
    return pd.DataFrame(rows, columns=COLS).fillna("").astype(str)


def append(data_dir, new):
    """add rows (a DataFrame with the COLS columns); never edits or deletes what is there"""
    new = pd.DataFrame(new).reindex(columns=COLS, fill_value="").fillna("").astype(str)
    if new.empty:
        return
    if sa is None:
        p = _csv(data_dir)
        old = pd.read_csv(p, dtype=str).fillna("") if p.exists() else pd.DataFrame(columns=COLS)
        tmp = p.with_suffix(".tmp")
        pd.concat([old, new], ignore_index=True)[COLS].to_csv(tmp, index=False)
        os.replace(tmp, p)                                 # swap in only when the write finished
        return
    eng = _engine(database_url(data_dir), str(_csv(data_dir)))
    t = _table(sa.MetaData())
    with eng.begin() as c:                                 # all rows or none
        c.execute(t.insert(), new.to_dict("records"))


def describe(data_dir):
    """where the decisions are stored, with the password hidden"""
    if sa is None:
        return f"CSV file {_csv(data_dir).as_posix()} (install sqlalchemy to use a database)"
    url = sa.engine.make_url(database_url(data_dir))
    if url.drivername.startswith("sqlite"):
        return f"SQLite file {url.database}"
    return url.render_as_string(hide_password=True)
