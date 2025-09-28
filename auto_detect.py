"""
auto_detect.py
Robust schema + domain detection with safe type coercion.

Exports:
- coerce_types_and_profile(df) -> (df_coerced, profile_dict)
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ---------- internal helpers ----------

_DATETIME_NAME_HINTS = {
    "date", "time", "timestamp", "datetime",
    "order date", "ship date", "visit_time", "visit date",
    "created", "updated"
}

_DATE_RE = re.compile(
    r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
)

def _looks_like_datetime_name(col: str) -> bool:
    c = col.strip().lower()
    return any(h in c for h in _DATETIME_NAME_HINTS)

def _sample_looks_datey(series: pd.Series, sample: int = 100) -> bool:
    if series.empty:
        return False
    s = series.dropna().astype(str).head(sample)
    if s.empty:
        return False
    hits = sum(bool(_DATE_RE.match(x.strip())) for x in s)
    return hits >= max(3, int(0.2 * len(s)))  # at least some / ~20%

def _coerce_object_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Cast object→numeric iff ≥90% of non-null values are numeric-shaped (commas tolerated)."""
    s = series
    if s.dtype != "object":
        return s, False
    nn = s.dropna().astype(str).str.replace(",", "", regex=False).str.strip()
    if nn.empty:
        return s, False
    maybe = nn.str.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", case=False).fillna(False)
    share = maybe.mean() if len(maybe) else 0.0
    if share >= 0.9:
        out = pd.to_numeric(nn, errors="coerce")
        out = out.reindex(s.index)
        out.loc[s.isna()] = np.nan
        return out, True
    return s, False

def _safe_to_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Parse datetimes with pandas >=2.0 mixed-format support."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        non_null = series.notna()
        share = ((parsed.notna() & non_null).sum() / non_null.sum()) if non_null.any() else 0.0
        if share >= 0.7:
            return parsed, True
        return series, False
    except Exception:
        return series, False

def coerce_types_and_profile(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Coerces obvious datetimes & numerics, classifies columns, guesses a domain,
    and returns (df_coerced, profile).
    """
    out = df.copy()

    # Normalize strings to pandas 'string' dtype (nullable)
    for c in out.columns:
        if out[c].dtype == "object":
            try:
                out[c] = out[c].astype("string")
            except Exception:
                pass

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []

    # 1) Coercion
    for c in out.columns:
        s = out[c]
        coerced = False

        # Try datetime only if name/value suggests it
        if _looks_like_datetime_name(c) or _sample_looks_datey(s):
            s2, changed = _safe_to_datetime(s)
            if changed:
                out[c] = s2
                datetime_cols.append(c)
                coerced = True

        if not coerced:
            s2, changed = _coerce_object_numeric(out[c])
            if changed:
                out[c] = s2
                coerced = True

        # Classification
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            if c not in datetime_cols:
                datetime_cols.append(c)
        elif pd.api.types.is_numeric_dtype(out[c]):
            if c not in numeric_cols:
                numeric_cols.append(c)
        else:
            if c not in categorical_cols:
                categorical_cols.append(c)

    # 2) Domain guess
    cols_lower = {c.lower() for c in out.columns}
    def has_any(names: List[str]) -> bool:
        return any(n in cols_lower for n in names)

    domain = "Generic"
    if has_any(["sales", "profit", "order id", "order date", "ship date", "category", "sub-category"]):
        domain = "Sales"
    elif has_any(["patient_id", "bp_sys", "bp_dia", "heart_rate", "temperature_c", "visit_time"]):
        domain = "Medical"
    elif has_any(["device_id", "status", "battery_pct", "timestamp", "humidity_pct", "temperature_c"]):
        domain = "IoT"
    elif has_any(["student_id", "exam", "score", "attendance_rate", "gpa"]):
        domain = "Education"
    elif has_any(["review_text", "rating", "sentiment_label", "verified_purchase"]):
        domain = "Reviews"

    # 3) Column metadata
    meta = []
    n = len(out)
    for c in out.columns:
        null_pct = float(out[c].isna().mean()) if n else 0.0
        meta.append({"name": c, "dtype": str(out[c].dtype), "null_pct": round(null_pct * 100, 2)})

    profile = {
        "columns": len(out.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "columns_meta": meta,
        "domain": domain,
    }
    return out, profile



