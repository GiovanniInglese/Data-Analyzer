"""
ai_helper.py
Domain-aware summaries & quick insights.

Exports:
- build_executive_summary(df, profile) -> {
      "domain": <str>,
      "executive_summary": [ ... ],
      "insights": [ ... ]
  }
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd


def _fmt_money(x) -> str:
    try:
        return "${:,.0f}".format(float(x))
    except Exception:
        return "—"

def _pct(a, b):
    try:
        if b in (0, None) or pd.isna(b):
            return None
        return 100.0 * float(a) / float(b)
    except Exception:
        return None

# -------- Sales --------
def _summ_sales(df: pd.DataFrame) -> List[str]:
    bullets: List[str] = []

    if "Sales" in df.columns:
        total = df["Sales"].sum(skipna=True)
        bullets.append(f"Total revenue: {_fmt_money(total)}.")

        if "Order Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Order Date"]):
            monthly = (
                df.dropna(subset=["Order Date"])
                  .assign(_ym=lambda d: d["Order Date"].dt.to_period("M").dt.to_timestamp())
                  .groupby("_ym")["Sales"].sum()
                  .sort_index()
            )
            if len(monthly) >= 3:
                x = np.arange(len(monthly))
                y = monthly.values.astype(float)
                slope = np.polyfit(x, y, 1)[0]
                bullets.append(f"Trend slope: {_fmt_money(slope)} per month.")

                direction = "declining" if slope < 0 else "growing"
                bullets.append(f"Revenue is {direction} month-over-month based on the linear trend.")


                k = 3
                if len(monthly) >= 2 * k:
                    last = monthly.iloc[-k:].sum()
                    prev = monthly.iloc[-2*k:-k].sum()
                    pct = _pct(last - prev, prev)
                    if pct is not None:
                        bullets.append(f"Momentum (last {k} vs prior {k}): {pct:.1f}%.")

        # Top driver
        for col in ("Category", "Region", "Sub-Category", "Segment"):
            if col in df.columns:
                top = df.groupby(col)["Sales"].sum().sort_values(ascending=False).head(2)
                if len(top):
                    parts = ", ".join([f"{idx} ({_fmt_money(val)})" for idx, val in top.items()])
                    bullets.append(f"Top {col}: {parts}.")
                break

    return bullets or ["Sales dataset detected; add Sales and Order Date for richer KPIs."]

# -------- Medical --------
def _summ_medical(df: pd.DataFrame) -> List[str]:
    bullets: List[str] = []

    if "bp_sys" in df.columns and pd.api.types.is_numeric_dtype(df["bp_sys"]):
        m = float(np.nanmean(df["bp_sys"]))
        if m >= 130:
            bullets.append(f"Average systolic BP is elevated at {m:.0f} mmHg (≥130).")

    if "bp_dia" in df.columns and pd.api.types.is_numeric_dtype(df["bp_dia"]):
        m = float(np.nanmean(df["bp_dia"]))
        if m >= 80:
            bullets.append(f"Average diastolic BP is high at {m:.0f} mmHg (≥80).")

    if "heart_rate" in df.columns and pd.api.types.is_numeric_dtype(df["heart_rate"]):
        m = float(np.nanmean(df["heart_rate"]))
        if m >= 100:
            bullets.append(f"Tachycardia tendency detected (mean ~{m:.0f} bpm).")

    if "temperature_c" in df.columns and pd.api.types.is_numeric_dtype(df["temperature_c"]):
        mx = float(np.nanmax(df["temperature_c"]))
        if mx >= 38.0:
            bullets.append(f"Possible fever cases observed (max {mx:.1f}°C).")

    if "visit_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df.get("visit_time")):
        by_day = (
            df.dropna(subset=["visit_time"])
              .assign(_d=lambda d: d["visit_time"].dt.to_period("D").dt.to_timestamp())
              .groupby("_d")
              .size()
              .sort_values(ascending=False)
              .head(2)
        )
        if len(by_day):
            parts = ", ".join([f"{idx.date()} ({val:,})" for idx, val in by_day.items()])
            bullets.append(f"Peak visit days: {parts}.")

    return bullets or ["Medical dataset loaded; no clear risks from aggregate stats."]

# -------- IoT --------
def _summ_iot(df: pd.DataFrame) -> List[str]:
    bullets: List[str] = []

    if "status" in df.columns:
        counts = df["status"].value_counts(dropna=True)
        total = int(counts.sum())
        if total:
            fail = int(counts.get("FAIL", 0))
            warn = int(counts.get("WARN", 0))
            bullets.append(f"Device status: FAIL={fail} ({(fail/total)*100:.1f}%), WARN={warn} ({(warn/total)*100:.1f}%).")

    if "battery_pct" in df.columns and pd.api.types.is_numeric_dtype(df["battery_pct"]):
        low = int((df["battery_pct"] < 20).sum())
        bullets.append(f"Low-battery readings (<20%): {low:,}.")

    if {"timestamp", "temperature_c"}.issubset(df.columns):
        by_hour = (
            df.dropna(subset=["timestamp"])
              .assign(_h=lambda d: d["timestamp"].dt.floor("H"))
              .groupby("_h")["temperature_c"]
              .mean()
        )
        if len(by_hour) >= 24:
            bullets.append("Hourly temperature signal available for seasonality/alerts.")
    return bullets or ["IoT dataset loaded; no critical aggregate alerts."]

# -------- Education --------
def _summ_education(df: pd.DataFrame) -> List[str]:
    bullets: List[str] = []
    if "score" in df.columns and pd.api.types.is_numeric_dtype(df["score"]):
        m = float(np.nanmean(df["score"]))
        pass_rate = float(np.mean(df["score"] >= 60) * 100)
        bullets.append(f"Average score: {m:.1f}; pass rate: {pass_rate:.1f}%.")

    if "attendance_rate" in df.columns and pd.api.types.is_numeric_dtype(df["attendance_rate"]):
        r = float(np.nanmean(df["attendance_rate"]) * 100)
        bullets.append(f"Average attendance: {r:.1f}%.")

    if "gpa" in df.columns and pd.api.types.is_numeric_dtype(df["gpa"]):
        g = float(np.nanmean(df["gpa"]))
        bullets.append(f"Mean GPA: {g:.2f}.")
    return bullets or ["Education dataset loaded; add score/attendance/GPA for richer insights."]

# -------- Reviews --------
def _summ_reviews(df: pd.DataFrame) -> List[str]:
    bullets: List[str] = []
    if "rating" in df.columns and pd.api.types.is_numeric_dtype(df["rating"]):
        avg = float(np.nanmean(df["rating"]))
        n = int(df["rating"].notna().sum())
        bullets.append(f"Average rating: {avg:.2f} across {n:,} reviews.")
    if "sentiment_label" in df.columns and "rating" in df.columns:
        n = int(df["rating"].notna().sum())
        neg = int((df["sentiment_label"] == "negative").sum())
        if n:
            bullets.append(f"Negative share: {(neg / n) * 100:.1f}%.")
    if "verified_purchase" in df.columns:
        vp = float(np.mean(df["verified_purchase"]) * 100)
        bullets.append(f"Verified purchases: {vp:.1f}% of reviews.")
    return bullets or ["Review dataset loaded; add ratings/sentiment for richer insights."]

# -------- main entry --------
def build_executive_summary(df: pd.DataFrame, profile: Dict) -> Dict:
    domain = (profile or {}).get("domain", "Generic")

    if domain == "Sales":
        bullets = _summ_sales(df)
    elif domain == "Medical":
        bullets = _summ_medical(df)
    elif domain == "IoT":
        bullets = _summ_iot(df)
    elif domain == "Education":
        bullets = _summ_education(df)
    elif domain == "Reviews":
        bullets = _summ_reviews(df)
    else:
        # Generic fallback
        n, m = df.shape
        null_share = float(df.isna().mean().mean()) * 100 if n and m else 0.0
        bullets = [f"Dataset shape: {n:,} rows × {m:,} columns.",
                   f"Average missingness across columns: {null_share:.1f}%."]

    addl: List[str] = []
    if "Sales" in df.columns and "Profit" in df.columns and df["Sales"].sum(skipna=True) != 0:
        margin = df["Profit"].sum(skipna=True) / df["Sales"].sum(skipna=True)
        addl.append(f"Overall margin: {margin*100:.1f}%.")

    return {"domain": domain, "executive_summary": bullets, "insights": addl}
