# CareGuardian synthetic demonstration

**Dataset:** 576 invented dose events from
36 fictional participants.

- Observed dose-event completion: 54.0%
- Group-held-out logistic regression accuracy: 0.597
- Group-held-out recall: 0.797
- Simulated inventory EOQ: 1039.23 units
- Reminder text patterns flagged for human review: 1

## Limitations

All people, doses, outcomes and financial inputs are artificial. The model is
a learning demonstration, not a validated medical device or adherence
recommendation engine. Reminder checks are simplistic string matching and
must not be used to determine clinical safety.
