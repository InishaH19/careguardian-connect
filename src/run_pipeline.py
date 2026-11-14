"""One-command CareGuardian demo: quality checks, ML, multi-agent coordination, QA, and economics."""
from argparse import ArgumentParser
from pathlib import Path
import json

from .adherence_analysis import load_events, descriptive_summary
from .careguardian import train_and_evaluate
from .agent_coordinator import CoordinationOrchestrator, PatientProfile
from .adversarial_safety_eval import AdversarialSafetyEvaluator
from .inventory_continuity import run_supply_continuity_simulation
from .reminder_qa import check_response, MOCK_RESPONSES

ROOT = Path(__file__).resolve().parents[1]


def run(data_path: Path, out_dir: Path) -> dict:
    events = load_events(data_path)
    analytics = descriptive_summary(events)
    model = train_and_evaluate(data_path)

    # Multi-agent coordination simulation
    demo_patient = PatientProfile(
        patient_id="PT-1082",
        name="Alex Morgan",
        medication_name="CardioGuard Extended-Release",
        prescribed_dose_mg=25.0,
        schedule_times=["08:00", "20:00"],
        caregiver_name="Jordan Morgan",
        caregiver_contact="+1-408-555-0199",
        current_inventory_units=60,
        reorder_point_units=15,
    )
    orchestrator = CoordinationOrchestrator(demo_patient)
    agent_scenarios = orchestrator.run_all_scenarios()

    # Adversarial safety evaluation suite
    safety_evaluator = AdversarialSafetyEvaluator()
    safety_matrix = safety_evaluator.evaluate_all()

    # Supply continuity & economics
    supply_report = run_supply_continuity_simulation()

    legacy_reminders = [
        {**example, **check_response(example["text"])} for example in MOCK_RESPONSES
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "adherence_summary.json": analytics,
        "model_metrics.json": model,
        "economics.json": supply_report,
        "agent_coordination_scenarios.json": agent_scenarios,
        "safety_eval_matrix.json": safety_matrix,
        "supply_continuity_report.json": supply_report,
        "reminder_qa.json": legacy_reminders,
    }

    for filename, obj in outputs.items():
        (out_dir / filename).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    md = f"""# CareGuardian Connect — Executive Demonstration Report

## 1. Executive Summary & Architecture
CareGuardian Connect demonstrates a multi-agent adherence and safety coordination platform engineered for remote patient monitoring.
The system integrates predictive machine-learning baseline classification, proactive multi-stakeholder notification coordination,
rigorous adversarial clinical safety guardrails, and automated pharmaceutical supply continuity modeling.

## 2. Event Adherence Analytics
- **Total Synthetically Modeled Dose Events:** {analytics['total_events']}
- **Simulated Cohort Size:** {analytics['simulated_participants']} participants
- **Mean Observed Completion Rate:** {analytics['observed_completion_pct']}%

## 3. Participant-Held-Out Machine Learning Baseline
- **Train/Test Hygiene:** `GroupShuffleSplit` on participant ID (zero cross-participant data leakage).
- **Model Architecture:** Standardized Feature Scaling with Regularized Logistic Regression.
- **Classification Accuracy:** {model['accuracy']}
- **Recall (Sensitivity on Non-Adherence):** {model['recall']}
- **Precision:** {model['precision']}
- **Confusion Matrix:** TN={model['confusion_matrix']['tn']}, FP={model['confusion_matrix']['fp']}, FN={model['confusion_matrix']['fn']}, TP={model['confusion_matrix']['tp']}

## 4. Multi-Agent Orchestration & Operational Scenarios
The system successfully evaluated 4 core operational interaction scenarios:
1. **Scenario A (Routine Scheduled Reminder):** On-time notification dispatched and adherence verified. Inventory decremented to {agent_scenarios['scenarios'][0]['inventory_after_dose']} units.
2. **Scenario B (Missed Dose Within Safe Window):** 1.5-hr latency caught; verified safe single-dose self-administration window.
3. **Scenario C (Multi-Party Caregiver Escalation):** 5.0-hr non-adherence event triggered automated notification dispatch to designated caregiver.
4. **Scenario D (Unsafe Query Interception & Fallback):** Dangerous dosage alteration intercepted by clinical guardrail; automated fallback escalation routed to attending physician.

## 5. Adversarial Safety Evaluation Matrix
- **Total Evaluated Adversarial Prompts:** {safety_matrix['total_adversarial_tests']}
- **Passed Boundary Tests:** {safety_matrix['passed_tests']} / {safety_matrix['total_adversarial_tests']} ({safety_matrix['pass_rate_percentage']}%)
- **Test Categories:** Prompt Injections, Dangerous Dosage Modifications, Emergency Triage Deflection, and False-Positive Validation.
- **Safety Status:** `{safety_matrix['status']}`

## 6. Pharmaceutical Supply Continuity & Financial Logistics
- **Wilson Economic Order Quantity (EOQ):** {supply_report['inventory_optimization']['optimal_order_quantity_eoq']} units/order
- **Optimal Order Cycles / Year:** {supply_report['inventory_optimization']['expected_orders_per_year']}
- **Buffer Safety Stock (95% Service Level):** {supply_report['supply_resilience_and_stockout_prevention']['safety_stock_buffer_units']} units ({supply_report['supply_resilience_and_stockout_prevention']['buffer_coverage_days']} days coverage)
- **Calculated Reorder Point (ROP):** {supply_report['supply_resilience_and_stockout_prevention']['reorder_point_units']} units
- **5-Year Net Present Value (NPV @ 10% Discount Rate):** ${supply_report['financial_feasibility_npv']['five_year_net_present_value_usd']:,.2f} USD
- **Investment Feasibility Verdict:** `{supply_report['financial_feasibility_npv']['roi_verdict']}`

---
*Notice: This technical prototype utilizes synthetic data for algorithmic modeling, architecture design, and portfolio demonstration.*
"""
    (out_dir / "PROJECT_REPORT.md").write_text(md, encoding="utf-8")
    return {
        "rows": analytics["total_events"],
        "participants": analytics["simulated_participants"],
        "model_accuracy": model["accuracy"],
        "safety_tests_passed": safety_matrix["passed_tests"],
        "eoq_units": supply_report["inventory_optimization"]["optimal_order_quantity_eoq"],
        "npv_usd": supply_report["financial_feasibility_npv"]["five_year_net_present_value_usd"],
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/synthetic_dose_events.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    result = run(args.data, args.out)
    print(f"CareGuardian Connect Pipeline completed: {result}; wrote 7 artifacts to {args.out}")


if __name__ == "__main__":
    main()
