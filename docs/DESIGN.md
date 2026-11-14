# Technical design and validation

## Data contract

Each row in `synthetic_dose_events.csv` describes one fabricated dose event
for one fabricated participant. `subject_id` is a synthetic grouping key, and
`day` identifies a numbered sample event/day. `day_of_week` (0–6),
`reminder_sent` (0 or 1), `prior_7d_adherence` (0–1), `doses_per_day` (positive
integer), and `travel_day` (0/1) are candidate features.
`took_dose` is a fabricated binary modeling target.

## Data flow

CSV > integrity validation > descriptive group statistics > per-participant
train/test partition > standardized logistic regression > model metrics.
In a separate deterministic branch: fixed mock text > phrase checks > human
review flags. In another branch: explicit invented finance inputs > EOQ/NPV.

## ML design rationale

Grouping by `subject_id` avoids the easy mistake of training and evaluating
on different events from the same simulated person. It does not eliminate
all leakage: the data-generation process, feature choices, preprocessing
windows and sample dependence would require scrutiny for any real use.
`StandardScaler` lives inside the training pipeline, so its parameters are
derived from the training partition. Accuracy alone is insufficient for rare
events; precision, recall and confusion-matrix cells are included.

## Error behavior

Malformed files raise `ValueError` with specific checks where possible.
The pipeline does not automatically impute missing data, so failures are
visible. No network, external credentials or runtime data collection occurs.

## Safety and ethics

The phrase heuristic is not a medication safety screen. Do not feed in
patient data or rely on reminder outputs for dosing. The project purposely
does not take action, dispense a pill or give treatment guidance.

## Validation plan

Use a deliberately duplicated `(subject_id, day)` row to verify rejection,
remove a required column, verify a probability outside 0–1 fails, inspect
grouped metrics and audit that every report comes from the included CSV.
For improvements, add seeded k-fold group cross validation, calibration,
a descriptive data card and sensitivity analyses.
