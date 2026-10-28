"""small helpers with no third-party dependency"""
import numpy as np
import pandas as pd


def md_table(df, digits=3):
    """a DataFrame as a Markdown table (no 'tabulate' package needed)"""
    if df is None or len(df) == 0:
        return "(none)"
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v).replace("|", "/").replace("\n", " ")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
