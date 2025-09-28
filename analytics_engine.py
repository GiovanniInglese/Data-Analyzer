"""
Flexible, domain-aware analytics — patched.
- Unifies cleaning: null tokens, percent strings, commas, winsorization
- Uses auto_detect.coerce_types_and_profile for robust typing & domain
- Uses ai_helper.build_executive_summary for narrative bullets
- Produces a single payload used directly by templates
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from auto_detect import coerce_types_and_profile   # profile: cols, types, domain, meta
from ai_helper import build_executive_summary      # domain-aware narratives

# ---------- Config ----------
MEASURE_HINTS = [
    "sales","revenue","amount","price","profit","gmv","net","subtotal","total","cost",
    "score","grade","attendance",
    "patients","visits","count","cases",
    "temperature","temp","value","reading"
]
DIM_PRIORITIES = [
    "segment","category","sub-category","sub_category","region","state","department",
    "class","grade","diagnosis","gender","age_group","city","country","ship mode"
]
NULL_TOKENS = {"", "na", "n/a", "null", "none", "-", "—", "n.a.", "nan"}

# ---------- Cleaning ----------
def _normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common null-like tokens to pd.NA (safe for object/string)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object" or str(out[c].dtype).startswith("string"):
            s = out[c].astype("string")
            mask = s.str.strip().str.lower().isin(NULL_TOKENS)
            # write back dtype-preserving
            s = s.mask(mask, pd.NA)
            out[c] = s
    return out

def _coerce_percent_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Turn '23%' -> 0.23 where a majority of non-null values end with %."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object" or str(out[c].dtype).startswith("string"):
            s = out[c].astype("string")
            mask = s.str.endswith("%", na=False)
            if mask.mean() >= 0.5 and mask.any():
                out[c] = pd.to_numeric(s.str.rstrip("%"), errors="coerce") / 100.0
    return out

def _strip_commas_and_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """If ≥70% of non-null values are numeric-shaped (after comma removal), coerce."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object" or str(out[c].dtype).startswith("string"):
            s = out[c].astype("string").str.replace(",", "", regex=False).str.strip()
            pattern = r"^[-+]?\d*\.?\d+(e[-+]?\d+)?$"
            looks = s.str.match(pattern, case=False, na=False)
            if looks.mean() >= 0.7:
                out[c] = pd.to_numeric(s, errors="coerce")
    return out

def _winsorize_series(s: pd.Series, p_low=0.01, p_high=0.99) -> pd.Series:
    ql, qh = s.quantile([p_low, p_high])
    return s.clip(lower=ql, upper=qh)

def _winsorize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]) and out[c].nunique(dropna=True) > 12:
            out[c] = _winsorize_series(pd.to_numeric(out[c], errors="coerce"))
    return out

def _pick_datetime(dfc: pd.DataFrame, profile: Dict) -> Optional[str]:
    dt_cols = list((profile or {}).get("datetime_cols", []))
    if dt_cols:
        return dt_cols[0]
    candidates = [c for c in dfc.columns if any(k in str(c).lower() for k in ("date","time","timestamp"))]
    return candidates[0] if candidates else None

# ---------- Detection helpers ----------
def _numeric_cols(df): return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def _categorical_cols(df):
    cats = []
    n = max(len(df), 1)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        u = df[c].nunique(dropna=True)
        if 1 < u <= max(50, int(0.3 * n)):
            cats.append(c)
    return cats

def _choose_measure(df, nums):
    # Prefer businessy columns if present
    preferred = ["Sales","Revenue","GMV","Amount","Total","Profit","Net","Subtotal","Cost"]
    for p in preferred:
        if p in df.columns and pd.api.types.is_numeric_dtype(df[p]):
            return p
    # Then fall back to hints / variance
    for c in nums:
        if any(h in str(c).lower() for h in MEASURE_HINTS):
            return c
    if nums:
        var = {c: float(np.nanvar(pd.to_numeric(df[c], errors="coerce"))) for c in nums}
        return max(var, key=var.get)
    return None


def _choose_dims(cats):
    if not cats:
        return []
    def score(col: str):
        low = str(col).lower()
        hit = -DIM_PRIORITIES.index(low) if low in DIM_PRIORITIES else 0
        return (hit, low)
    return sorted(cats, key=score, reverse=True)[:2]

# ---------- KPIs ----------
def _fmt_measure(val: float) -> float:
    return float(val)

def _kpis(dfc: pd.DataFrame, dt_col: Optional[str], measure: Optional[str]) -> Dict[str, Any]:
    k = {"rows": int(len(dfc)), "columns": int(dfc.shape[1])}
    if dt_col and dt_col in dfc.columns and pd.api.types.is_datetime64_any_dtype(dfc[dt_col]):
        k["date_min"] = str(dfc[dt_col].min())
        k["date_max"] = str(dfc[dt_col].max())
    if measure and measure in dfc.columns and pd.api.types.is_numeric_dtype(dfc[measure]):
        s = pd.to_numeric(dfc[measure], errors="coerce").fillna(0.0)
        k[f"total_{measure}"] = _fmt_measure(s.sum())
        k[f"avg_{measure}"] = _fmt_measure(s.mean())
        if dt_col and dt_col in dfc.columns and pd.api.types.is_datetime64_any_dtype(dfc[dt_col]):
            monthly = (dfc.set_index(dt_col).groupby(pd.Grouper(freq="MS"))[measure].sum().dropna())
            if len(monthly) > 0:
                k[f"avg_monthly_{measure}"] = _fmt_measure(monthly.mean())
            if len(monthly) >= 6:
                last3 = monthly.tail(3).mean()
                prev3 = monthly.tail(6).head(3).mean()
                if prev3 and not np.isclose(prev3, 0.0):
                    k["momentum_3mo_vs_prev3"] = float((last3/prev3 - 1.0) * 100.0)
            if len(monthly) >= 2:
                x = np.arange(len(monthly)); y = monthly.values.astype(float)
                k["trend_slope_per_month"] = float(np.polyfit(x, y, 1)[0])
                imax = int(np.argmax(y)); imin = int(np.argmin(y))
                k["best_month"] = str(monthly.index[imax])[:7]
                k["best_month_value"] = _fmt_measure(y[imax])
                k["worst_month"] = str(monthly.index[imin])[:7]
                k["worst_month_value"] = _fmt_measure(y[imin])
    return k

# ---------- Insights ----------
def _exec_actions(dfc, domain, dt_col, measure):
    acts = []
    if domain.lower() == "sales":
        for a, b in [("Discount","Profit"), ("Discount","Sales")]:
            if a in dfc.columns and b in dfc.columns:
                corr = dfc[[a,b]].corr().iloc[0,1]
                if not math.isnan(corr) and corr < -0.15:
                    acts.append(f"Higher {a} tends to reduce {b} (corr {corr:.2f}); review promotion policy.")
    if dt_col and (measure in dfc.columns if measure else False):
        monthly = (dfc.set_index(dt_col).groupby(pd.Grouper(freq="MS"))[measure].sum().dropna())
        if len(monthly) >= 6:
            last3 = monthly.tail(3).mean()
            prev3 = monthly.tail(6).head(3).mean()
            if prev3:
                d = (last3/prev3 - 1) * 100
                if d > 8:
                    acts.append("Momentum accelerating over the last quarter.")
                elif d < -8:
                    acts.append("Momentum slowing over the last quarter.")
    return acts or ["No immediate actions detected."]

def _descriptive_insights(dfc, dims, measure, domain, dt_col):
    tips = []
    if dt_col and dt_col in dfc.columns and pd.api.types.is_datetime64_any_dtype(dfc[dt_col]):
        s = dfc[dt_col].dt.floor("D")
        agg = s.value_counts().sort_values(ascending=False).head(2)
        if not agg.empty:
            pairs = ", ".join([f"{idx.date()} ({val:,.0f})" for idx, val in agg.items()])
            tips.append(f"Top date volume: {pairs}")
    if dims:
        col = dims[0]
        if measure and measure in dfc.columns and pd.api.types.is_numeric_dtype(dfc[measure]):
            agg = dfc.groupby(col)[measure].sum().sort_values(ascending=False).head(2)
            pair = ", ".join([f"{idx}: {val:,.0f}" for idx, val in agg.items()])
            tips.append(f"Top {col} by {measure}: {pair}")
        else:
            agg = dfc[col].value_counts().head(2)
            pair = ", ".join([f"{idx}: {val:,.0f}" for idx, val in agg.items()])
            tips.append(f"Top {col} by count: {pair}")
    return tips

# ---------- Charts ----------
def _kpi_series(dfc, dt_col, measure):
    if not dt_col or dt_col not in dfc.columns or not pd.api.types.is_datetime64_any_dtype(dfc[dt_col]):
        return None
    if measure and measure in dfc.columns and pd.api.types.is_numeric_dtype(dfc[measure]):
        s = (dfc.set_index(dt_col).groupby(pd.Grouper(freq="MS"))[measure].sum().dropna())
        title = f"{measure} by Month (Total)"; ylab = measure
    else:
        s = (dfc.set_index(dt_col).groupby(pd.Grouper(freq="MS")).size().astype(float))
        title = "Records by Month (Count)"; ylab = "Count"
    if len(s) == 0:
        return None
    return {"title": title, "y_label": ylab,
            "labels": [str(p)[:7] for p in s.index],
            "values": [float(v) for v in s.values]}

def _segments(dfc, dims, measure):
    arr = []
    for col in dims[:2]:
        if col not in dfc.columns:
            continue
        if measure and measure in dfc.columns and pd.api.types.is_numeric_dtype(dfc[measure]):
            agg = dfc.groupby(col)[measure].sum().sort_values(ascending=False).head(10)
            label = "Total"
        else:
            agg = dfc.groupby(col).size().sort_values(ascending=False).head(10)
            label = "Count"
        arr.append({
            "column": col,
            "metric_label": label,
            "top_categories": [{"category": str(i), "metric": float(v)} for i, v in agg.items()]
        })
    return arr

def _numeric_histograms(dfc, numeric_cols) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for col in numeric_cols[:10]:
        try:
            s = pd.to_numeric(dfc[col], errors="coerce").dropna()
            if len(s) < 5:
                continue
            bins = min(12, max(6, int(round(np.sqrt(len(s))))))
            counts, edges = np.histogram(s.values, bins=bins)
            labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges)-1)]
            out[f"{col}_bins"] = labels
            out[f"{col}_counts"] = [int(x) for x in counts.tolist()]
        except Exception:
            continue
    return out

# ---------- Public API ----------
def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Main entry for Flask route. Cleans, profiles, builds KPIs, charts, and narratives."""
    # Stage 1: normalize obvious issues before profiling
    df1 = _normalize_nulls(df)
    df1 = _coerce_percent_strings(df1)
    df1 = _strip_commas_and_numeric(df1)

    # Stage 2: robust coercion + domain/profile via auto_detect
    dfc, profile = coerce_types_and_profile(df1)

    # Optional: winsorize long-tailed numerics for stabler charts
    dfc = _winsorize_numeric(dfc)

    # Assemble pieces
    dt_col = _pick_datetime(dfc, profile)
    nums = _numeric_cols(dfc)
    cats = _categorical_cols(dfc)
    measure = _choose_measure(dfc, nums)
    dims = _choose_dims(cats)
    domain = (profile or {}).get("domain", "Generic")

    kpis = _kpis(dfc, dt_col, measure)
    exec_actions = _exec_actions(dfc, domain, dt_col, measure)
    desc_insights = _descriptive_insights(dfc, dims, measure, domain, dt_col)

    # Narrative bullets (domain-aware)
    summaries = build_executive_summary(dfc, profile)

    payload: Dict[str, Any] = {
        "kpis": kpis,
        "executive_summary": summaries.get("executive_summary", []) or exec_actions,
        "insights": (summaries.get("insights", []) or []) + desc_insights,
        "charts": {
            "kpi_series": _kpi_series(dfc, dt_col, measure),
            "segments": _segments(dfc, dims, measure)
        },
        "profile": {
            "columns": dfc.shape[1],
            "columns_meta": [{"name": c,
                              "dtype": str(dfc[c].dtype),
                              "null_pct": float(dfc[c].isna().mean()*100)} for c in dfc.columns],
            "measure": measure,
            "dims": dims,
            "domain": domain,
            "datetime_col": dt_col,
            "numeric_cols": nums,
            "categorical_cols": cats
        }
    }

    # Add numeric histograms for templates that expect <col>_bins / <col>_counts
    payload.update(_numeric_histograms(dfc, nums))

    return payload

# Backward-compat alias for older imports
build_results_payload = analyze_dataset




