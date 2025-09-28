# insights/data_utils.py
import pandas as pd

# Use non-GUI backend BEFORE pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import numpy as np



def summarize_data(df: pd.DataFrame):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": {str(c): int(m) for c, m in df.isnull().sum().items()}
    }

# app.py or data_utils.py — wherever you build the analysis payload

import numpy as np
import pandas as pd

def add_histograms(payload, df, numeric_cols, bins=20):
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) >= 5:
            counts, edges = np.histogram(s, bins=bins)
            payload[f"{c}_bins"] = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges)-1)]
            payload[f"{c}_counts"] = counts.tolist()
        else:
            payload[f"{c}_bins"] = []
            payload[f"{c}_counts"] = []
    return payload


def kpi_sales(df: pd.DataFrame):
    if "sales" in df.columns:
        s = pd.to_numeric(df["sales"], errors="coerce")
        return {
            "total_sales": float(s.sum(skipna=True)),
            "avg_sales": float(s.mean(skipna=True))
        }
    return {}

def _maybe_sample(df: pd.DataFrame, max_rows: int = 200_000) -> pd.DataFrame:
    n = len(df)
    if n > max_rows:
        return df.sample(max_rows, random_state=42)
    return df

def plot_sales_by_category(df: pd.DataFrame, save_path="static/sales_by_category.png"):
    if "category" in df.columns and "sales" in df.columns:
        df = df.copy()
        df = _maybe_sample(df, 200_000)

        df["category"] = df["category"].astype(str)
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

        cat_sales = df.groupby("category", dropna=True)["sales"].sum().sort_values(ascending=False)
        if cat_sales.empty:
            return None
        cat_sales = cat_sales.head(15)

        plt.figure(figsize=(7.5, 4.5))
        cat_sales.plot(kind="bar")
        plt.ylabel("Total Sales")
        plt.title("Total Sales by Category (Top 15)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
    return None

def plot_sales_over_time(df: pd.DataFrame, save_path="static/sales_over_time.png"):
    if "order_date" in df.columns and "sales" in df.columns:
        df = df.copy()
        df = _maybe_sample(df, 300_000)

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        daily = df.dropna(subset=["order_date"]).groupby("order_date")["sales"].sum().sort_index()
        if daily.empty:
            return None

        if len(daily) > 2000:
            daily = daily.iloc[:: max(1, len(daily)//2000)]

        plt.figure(figsize=(7.5, 4.5))
        plt.plot(daily.index, daily.values)
        plt.ylabel("Sales")
        plt.title("Sales Over Time")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
    return None

def compact_context(df: pd.DataFrame, max_cats: int = 5):
    ctx = {}

    if {"category", "sales"}.issubset(df.columns):
        t = df.copy()
        t["sales"] = pd.to_numeric(t["sales"], errors="coerce")
        cat_sales = (
            t.groupby("category")["sales"]
             .sum()
             .sort_values(ascending=False)
             .head(max_cats)
             .round(2)
        )
        ctx["top_categories"] = {str(k): float(v) for k, v in cat_sales.items()}

    if {"order_date", "sales"}.issubset(df.columns):
        d = df.copy()
        d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
        d["sales"] = pd.to_numeric(d["sales"], errors="coerce")
        daily = d.dropna(subset=["order_date"]).groupby("order_date")["sales"].sum().sort_index()
        tail = daily.tail(10).round(2)
        ctx["recent_days"] = {str(k.date()): float(v) for k, v in tail.items()}

        if len(daily) >= 6:
            last3 = daily.tail(3).mean()
            prev3 = daily.tail(6).head(3).mean()
            if pd.notna(prev3) and prev3 != 0:
                delta_pct = (last3 - prev3) / prev3 * 100
            else:
                delta_pct = float("inf") if pd.notna(last3) and last3 > 0 else 0.0
            ctx["recent_trend_pct"] = round(float(delta_pct), 2)

    return ctx

# --- Forecast: simple linear regression over daily totals ---
def forecast_sales(df: pd.DataFrame, days_ahead: int = 7, save_path="static/sales_forecast.png"):
    """
    Robust forecast:
    - Auto-detect date and sales columns (common variants).
    - Coerce sales to numeric.
    - Works with tiny datasets:
        * 0 valid days  -> (None, None)
        * 1 day         -> flat forecast (repeat last value)
        * 2–9 days      -> linear regression
        * >=10 days     -> linear regression (same as above)
    """
    # --- 1) Detect column names (case-insensitive) ---
    cols = {c.lower(): c for c in df.columns.astype(str)}
    date_candidates  = ["order_date", "date", "orderdate", "order date", "orderday"]
    sales_candidates = ["sales", "revenue", "amount", "total", "total_sales", "gross"]

    date_col  = next((cols[k] for k in date_candidates  if k in cols), None)
    sales_col = next((cols[k] for k in sales_candidates if k in cols), None)
    if not date_col or not sales_col:
        return None, None  # can't forecast without date & sales

    # --- 2) Clean types ---
    d = df.copy()
    d[date_col]  = pd.to_datetime(d[date_col], errors="coerce")
    d[sales_col] = pd.to_numeric(
        d[sales_col]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True),  # strip $ and commas etc.
        errors="coerce"
    )

    # --- 3) Aggregate to daily totals ---
    daily = (
        d.dropna(subset=[date_col])
         .groupby(d[date_col].dt.date)[sales_col]
         .sum()
         .sort_index()
    )

    n = len(daily)
    if n == 0:
        return None, None

    # --- 4) Choose strategy by data size ---
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    # Make an index for modeling
    y = daily.values
    X = np.arange(n).reshape(-1, 1)

    if n == 1:
        # Flat forecast: repeat the single known value
        last_val = float(y[-1])
        future_vals = np.full(days_ahead, last_val, dtype=float)
    else:
        # Fit a simple linear regression for any n >= 2
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(n, n + days_ahead).reshape(-1, 1)
        future_vals = model.predict(future_X).astype(float)

    # --- 5) Build future date index ---
    last_date = pd.to_datetime(daily.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
    forecast_series = pd.Series(np.round(future_vals, 2), index=future_dates)

    # --- 6) Plot (historical + forecast) ---
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(pd.to_datetime(list(daily.index)), y, label="Historical Sales")
    plt.plot(forecast_series.index, forecast_series.values, "r--", label="Forecast")
    plt.title(f"Sales Forecast (Next {days_ahead} Days)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {str(k.date()): float(v) for k, v in forecast_series.items()}, save_path

