"""Descriptive analysis of fictional medication events; not clinical inference.

The analytics module deliberately avoids making patient-level recommendations.
It performs data-quality checks and computes group-level metrics that can be
audited against the included CSV file.
"""
from pathlib import Path
import pandas as pd

REQUIRED = {
    "subject_id", "day", "day_of_week", "reminder_sent",
    "prior_7d_adherence", "doses_per_day", "travel_day", "took_dose",
}

def load_events(path: Path) -> pd.DataFrame:
    """Load and validate a synthetic dose-event table.

    Reject missing columns, empty participant IDs, duplicate (person, day)
    records, impossible probabilities, and non-binary indicators. The function
    never silently drops suspicious rows or imputes unknown information.
    """
    events = pd.read_csv(path)
    missing = REQUIRED - set(events)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if events.empty or events[list(REQUIRED)].isna().any().any():
        raise ValueError("Dataset must be nonempty without missing required values")
    if events["subject_id"].astype(str).str.strip().eq("").any():
        raise ValueError("subject_id cannot be empty")
    if events.duplicated(subset=["subject_id", "day"]).any():
        raise ValueError("Duplicate participant/day event")
    for field in ("reminder_sent", "travel_day", "took_dose"):
        if not events[field].isin([0, 1]).all():
            raise ValueError(f"{field} must contain only 0 or 1")
    if not events["prior_7d_adherence"].between(0, 1).all():
        raise ValueError("Prior adherence values must be between zero and one")
    if not events["day_of_week"].between(0, 6).all():
        raise ValueError("day_of_week must be between 0 and 6")
    if not (events["doses_per_day"] >= 1).all():
        raise ValueError("doses_per_day must be at least one")
    return events

def descriptive_summary(events: pd.DataFrame) -> dict:
    """Summarize aggregate dose-event outcomes without causal assertions."""
    if events.empty:
        raise ValueError("Cannot analyze an empty dataset")
    reminder = events.groupby("reminder_sent")["took_dose"].agg(["mean", "count"])
    reminder_breakdown = {
        ("reminder" if int(code) == 1 else "no_reminder"): {
            "events": int(row["count"]),
            "observed_completion_pct": round(float(row["mean"] * 100), 1),
        }
        for code, row in reminder.iterrows()
    }
    participant_rates = events.groupby("subject_id")["took_dose"].mean()
    return {
        "dataset": "synthetic teaching data; not real patient observations",
        "total_events": len(events),
        "simulated_participants": int(events["subject_id"].nunique()),
        "observed_completion_pct": round(float(events["took_dose"].mean() * 100), 1),
        "participant_completion_pct": {
            "min": round(float(participant_rates.min() * 100), 1),
            "mean": round(float(participant_rates.mean() * 100), 1),
            "max": round(float(participant_rates.max() * 100), 1),
        },
        "reminder_breakdown": reminder_breakdown,
        "interpretation": (
            "Descriptive association only. This synthetic dataset cannot establish "
            "that a reminder causes adherence or supports clinical decisions."
        ),
    }
