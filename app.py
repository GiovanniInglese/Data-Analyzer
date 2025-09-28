# app.py — BizInsights (Flask + Pandas + Chart.js) — drop-in with unified reader
from __future__ import annotations

import io
import os
import traceback
from typing import Dict, Any

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
)
import pandas as pd

# Your analyzer
from analytics_engine import analyze_dataset

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret"),
    MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100 MB upload cap
    UPLOAD_FOLDER=os.environ.get("UPLOAD_DIR", os.path.join(os.getcwd(), "uploads")),
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".xml"}  # feather intentionally omitted

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in ALLOWED_EXTS

def _sniff_csv_delimiter(sample: bytes) -> str:
    """
    Very light delimiter sniffing for CSV/TSV when users upload odd files.
    Defaults to comma; falls back to tab or semicolon if those look dominant.
    """
    txt = sample.decode(errors="ignore")
    head = "\n".join(txt.splitlines()[:5])
    counts = {",": head.count(","), "\t": head.count("\t"), ";": head.count(";")}
    best = max(counts, key=counts.get)
    return "\t" if best == "\t" else (";" if best == ";" else ",")

def read_any_file(file_or_path) -> pd.DataFrame:
    """
    Read CSV / TSV / Excel / Parquet / JSON / XML with graceful fallbacks.
    - Parquet loads via engine="auto" (fastparquet in your env).
    - Feather intentionally not supported (pyarrow not installed).
    Accepts either a Werkzeug FileStorage or a filesystem path.
    """
    # Resolve a filesystem path
    if hasattr(file_or_path, "read"):  # FileStorage from request.files['file']
        incoming_name = getattr(file_or_path, "filename", "") or "upload"
        _, ext = os.path.splitext(incoming_name)
        ext = ext.lower()
        if not _allowed_file(incoming_name):
            raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

        # Save to uploads to ensure libraries that depend on extension can sniff it
        safe_name = incoming_name.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file_or_path.save(save_path)
        path = save_path
    else:
        path = str(file_or_path)
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if not _allowed_file(path):
            raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

    # Dispatch by extension
    if ext == ".csv":
        # Try fast path; if it fails (weird delimiter/encoding), sniff
        try:
            return pd.read_csv(path)
        except Exception:
            with open(path, "rb") as fh:
                sample = fh.read(64 * 1024)
            delim = _sniff_csv_delimiter(sample)
            return pd.read_csv(path, sep=delim, engine="python", encoding_errors="ignore")

    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", engine="python", encoding_errors="ignore")

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)  # openpyxl is installed

    if ext == ".parquet":
        # engine="auto" → uses fastparquet in your current environment
        return pd.read_parquet(path, engine="auto")

    if ext == ".json":
        # Try JSON Lines first; fall back to normal JSON
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.read_json(path)

    if ext == ".xml":
        return pd.read_xml(path)

    raise ValueError(f"Unsupported file type: {ext}")

def _safe_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure payload is JSON-serializable where needed (Jinja handles most cases).
    """
    # Pandas/Numpy types are usually serializable via Flask/Jinja tojson,
    # so we keep this light. Placeholders for future custom coercions.
    return d

def _friendly_error(e: Exception) -> str:
    msg = f"{type(e).__name__}: {e}"
    # Trim extremely long tracebacks from the UI; log full to console.
    app.logger.error("Error during analysis\n%s\n", traceback.format_exc())
    return msg

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/bench")
def bench():
    return render_template("bench.html")

@app.post("/analyze")
def analyze():
    file = request.files.get("file")
    if not file or not getattr(file, "filename", ""):
        return render_template("index.html", error="Please choose a file to upload."), 400
    try:
        df = read_any_file(file)
        if df is None or df.empty:
            return render_template("index.html", error="The uploaded file has no rows."), 400

        result = analyze_dataset(df)
        payload = _safe_payload(result)

        # Render dark results page with charts
        return render_template("results.html", result=payload)

    except Exception as e:
        return render_template("index.html", error=_friendly_error(e)), 400

@app.post("/api/analyze")
def api_analyze():
    """
    Optional JSON API: returns the analysis payload as JSON.
    """
    file = request.files.get("file")
    if not file or not getattr(file, "filename", ""):
        return jsonify({"error": "No file uploaded"}), 400
    try:
        df = read_any_file(file)
        if df is None or df.empty:
            return jsonify({"error": "Uploaded file has no rows"}), 400
        result = analyze_dataset(df)
        return jsonify(_safe_payload(result))
    except Exception as e:
        return jsonify({"error": _friendly_error(e)}), 400

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/favicon.ico")
def favicon():
    # Optional: serve a local favicon if you add one, avoid 404 noise in console
    favdir = os.path.join(os.getcwd(), "static")
    favpath = os.path.join(favdir, "favicon.ico")
    if os.path.exists(favpath):
        return send_from_directory(favdir, "favicon.ico")
    return ("", 204)

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # For local dev: `python app.py` or use `flask run`
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
