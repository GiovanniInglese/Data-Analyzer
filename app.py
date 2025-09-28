# app.py — robust loader + flexible engine import + correct 'index' endpoint
from __future__ import annotations

import io, csv, json, warnings, os
from typing import Any, Dict
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

# Flexible engine import (prefer our patched analytics_engine)
try:
    from insights import analyze_dataset  # optional thin wrapper
except Exception:
    try:
        from analytics_engine import analyze_dataset
    except Exception:
        from analytics_engine import build_results_payload as analyze_dataset  # backward compat

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev")

def _ext(filename: str) -> str:
    return (filename.rsplit(".", 1)[-1].lower() if "." in filename else "").strip()

def _sniff_sep(sample: bytes) -> str:
    try:
        text = sample.decode("utf-8", "ignore")
        dialect = csv.Sniffer().sniff(text, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return ","

def _load_json(buf: io.BytesIO) -> pd.DataFrame:
    buf.seek(0)
    raw = buf.read().decode("utf-8", "ignore").strip()
    if not raw:
        raise ValueError("Empty JSON file.")
    lines = raw.splitlines()
    # JSON lines?
    if len(lines) > 1 and sum(1 for ln in lines[:50] if ln.strip().startswith("{")) > 10:
        return pd.read_json(io.StringIO(raw), lines=True)
    # Array / object
    obj = json.loads(raw)
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return pd.json_normalize(v)
        return pd.json_normalize(obj)
    raise ValueError("Unsupported JSON structure.")

def _load_dataframe(file_storage) -> pd.DataFrame:
    """Reads uploaded file into a DataFrame (CSV/TSV/TXT/Excel/Parquet/Feather/JSON/XML)."""
    name = file_storage.filename or ""
    ext = _ext(name)
    raw = file_storage.read()
    if not raw:
        raise ValueError("Empty upload.")
    buf = io.BytesIO(raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if ext in ("csv", "txt", "tsv", ""):
            if ext == "tsv" or name.endswith(".tsv"):
                sep = "\t"
            else:
                buf.seek(0)
                sep = _sniff_sep(buf.read(8192))
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, engine="python")

        if ext in ("xlsx", "xls"):
            return pd.read_excel(buf)

        if ext == "parquet":
            return pd.read_parquet(buf)

        if ext == "feather":
            return pd.read_feather(buf)

        if ext == "json":
            return _load_json(buf)

        if ext == "xml":
            buf.seek(0)
            return pd.read_xml(buf)

        # Fallback: sniff and parse as CSV
        buf.seek(0)
        sep = _sniff_sep(buf.read(8192))
        buf.seek(0)
        return pd.read_csv(buf, sep=sep, engine="python")

@app.route("/", methods=["GET"], endpoint="index")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload_form():
    return render_template("upload.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        flash("Please choose a file to upload.", "warning")
        return redirect(url_for("index"))
    f = request.files["file"]
    if not f or not f.filename:
        flash("Please choose a file to upload.", "warning")
        return redirect(url_for("index"))
    try:
        df = _load_dataframe(f)
    except Exception as e:
        return f"<h3>File load failed</h3><pre>{e}</pre>"
    try:
        payload: Dict[str, Any] = analyze_dataset(df)
    except Exception as e:
        return f"<h3>Analyze failed</h3><pre>{e}</pre>"
    return render_template("results.html", result=payload)

@app.route("/bench", methods=["GET"])
def bench():
    return render_template("bench.html", results=[])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
