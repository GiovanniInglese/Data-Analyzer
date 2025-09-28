# AI-Powered Business Insights Dashboard (Step 1: MVP)

This is the starting scaffold for your vibe-coded Business Insights Dashboard.
In **Step 1**, you can upload a CSV and preview the first 10 rows in the browser.

## Prereqs
- Python 3.10+ installed
- Windows PowerShell or Terminal (VS Code works great)

## Setup (Windows PowerShell)
```powershell
# 1) Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the Flask app (on http://127.0.0.1:5000/)
python app.py
```

## Setup (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Use
1. Open your browser to http://127.0.0.1:5000/
2. Upload any CSV (a small sample is included in `uploads/sample_sales.csv`)
3. You'll see a preview table of the first 10 rows.

## Next Steps
- Add summary stats + charts
- Add AI insight generation
- Add basic predictive modeling
- Polish UI (filters, date ranges, KPIs)

---
This project is built with **vibe coding**: you describe the goal and iterate,
and the AI helps scaffold, debug, and extend. Your job is to guide and validate.