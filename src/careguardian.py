"""Synthetic data only. Demonstration, not clinical software."""
from pathlib import Path
import json
import math
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["day_of_week", "reminder_sent", "prior_7d_adherence",
            "doses_per_day", "travel_day"]

def economic_order_quantity(annual_demand: float, order_cost: float, holding_cost: float) -> float:
    """Basic inventory EOQ, for hypothetical assumptions only."""
    if min(annual_demand, order_cost, holding_cost) <= 0:
        raise ValueError("EOQ assumptions must be positive")
    return math.sqrt(2 * annual_demand * order_cost / holding_cost)

def net_present_value(discount_rate: float, initial_cost: float, annual_net_benefits: list[float]) -> float:
    if discount_rate <= -1:
        raise ValueError("Discount rate must exceed -100%")
    return -initial_cost + sum(
        benefit / (1 + discount_rate) ** year
        for year, benefit in enumerate(annual_net_benefits, start=1)
    )

def train_and_evaluate(csv_path: Path) -> dict:
    data = pd.read_csv(csv_path)
    required = {"subject_id", "took_dose"} | set(FEATURES)
    if required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data["subject_id"].nunique() < 5:
        raise ValueError("Need several simulated participants for a group split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_ix, test_ix = next(splitter.split(data[FEATURES], data["took_dose"],
                                            groups=data["subject_id"]))
    train, test = data.iloc[train_ix], data.iloc[test_ix]
    assert not set(train["subject_id"]) & set(test["subject_id"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[FEATURES], train["took_dose"])
    prediction = model.predict(test[FEATURES])
    tn, fp, fn, tp = confusion_matrix(test["took_dose"], prediction, labels=[0, 1]).ravel()
    return {
        "dataset_type": "fully synthetic; no actual patients",
        "target": "simulated took_dose",
        "train_rows": len(train), "test_rows": len(test),
        "train_subjects": train["subject_id"].nunique(),
        "test_subjects": test["subject_id"].nunique(),
        "accuracy": round(float(accuracy_score(test["took_dose"], prediction)), 3),
        "precision": round(float(precision_score(test["took_dose"], prediction, zero_division=0)), 3),
        "recall": round(float(recall_score(test["took_dose"], prediction, zero_division=0)), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "warning": "Not a medical device or a clinically validated adherence predictor."
    }

def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    metrics = train_and_evaluate(ROOT / "data/synthetic_dose_events.csv")
    economics = {
        "assumptions": "illustrative figures only; not based on Johnson & Johnson",
        "eoq_units": round(economic_order_quantity(36000, 45, 3), 1),
        "five_year_npv_usd": round(net_present_value(.10, 50000, [14000, 18000, 19000, 20000, 23000]), 2)
    }
    (reports / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (reports / "economics.json").write_text(json.dumps(economics, indent=2) + "\n")
    print("Saved reports/model_metrics.json and reports/economics.json")
    print(f"Synthetic grouped holdout — accuracy: {metrics['accuracy']}, recall: {metrics['recall']}")

if __name__ == "__main__":
    main()
