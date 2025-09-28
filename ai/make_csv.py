import pandas as pd
import numpy as np

def make_big_csv(path="big_business.csv", rows=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", periods=365)
    categories = np.array(["Electronics","Home","Grocery","Clothing","Beauty","Toys","Sports"])
    regions = np.array(["North","South","East","West"])
    customers = np.array([f"CUST-{i:05d}" for i in range(1, 5001)])

    dates = rng.choice(days, size=rows)
    cats = rng.choice(categories, size=rows, p=[0.22,0.18,0.20,0.15,0.10,0.08,0.07])
    regs = rng.choice(regions, size=rows)
    cust = rng.choice(customers, size=rows)

    base = rng.lognormal(mean=4.5, sigma=0.6, size=rows)
    multipliers = {
        "Electronics": 3.0, "Home": 1.8, "Grocery": 0.4, "Clothing": 0.9,
        "Beauty": 0.7, "Toys": 0.6, "Sports": 1.2
    }
    sales = (base * np.vectorize(multipliers.get)(cats)).round(2)
    qty = rng.integers(1, 12, size=rows)

    df = pd.DataFrame({
        "order_date": dates,
        "category": cats,
        "region": regs,
        "customer": cust,
        "sales": sales,
        "quantity": qty
    }).sort_values("order_date")

    df.to_csv(path, index=False)
    print(f"Wrote {rows:,} rows to {path}")

if __name__ == "__main__":
    make_big_csv(rows=2_000_000)  # change rows if you want bigger/smaller
